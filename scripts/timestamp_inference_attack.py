#!/usr/bin/env python3
"""
Global Timestamp Inference Attack for TKG models.

Threat model
------------
The attacker knows (h, r, o), but NOT the true timestamp t* and NOT a local
time window around it.

The attacker only knows the full temporal vocabulary supported by the model:

    T = {t1, t2, ..., tN}

and has black-box access to the model score:

    s(h, r, o, t)

For each target triple (h, r, o), the attacker queries ALL valid timestamps,
ranks them by score, and predicts the timestamp with the highest score.

Main privacy evaluation
-----------------------
- victim_train unique-time triples: privacy attack
- victim_test unique-time triples: held-out/generalization control

Metrics
-------
- Top@1
- Top@3
- Top@10
- MRR
- Mean Rank
- Median Rank
- Mean / Median temporal-step error of Top-1 prediction

Baselines
---------
- Random ranking
- Timestamp-frequency ranking learned from victim_train only

Supported models
----------------
- TeRo-like
- TComplEx
- TNTComplEx

Expected project structure
--------------------------
/home/harrouch/Paper2/Paper2/
  processed/<DATASET>/
      victim_train.txt
      victim_test.txt
      time2id.json
  models/<DATASET>/
      tero_victim.pt
      tcomplex_victim.pt
      tntcomplex_victim.pt

Author note
-----------
This implementation matches the scoring conventions used in the current
TKG privacy pipeline:
- TeRo-like: rotate h and o with time phase; add relation after rotation.
- TComplEx: Re(<h, r, conj(o), t>).
- TNTComplEx: r_eff(t) = r_time * t + r_static.

"""

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch


DEFAULT_ROOT = "/home/harrouch/Paper2/Paper2"

DEFAULT_DATASETS = [
    "ICEWS05-15",
    "ICEWS14",
    "YAGO11k",
    "WIKIDATA12k",
]

DEFAULT_MODELS = [
    "tero",
    "tcomplex",
    "tntcomplex",
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data loading
# ============================================================

def load_quadruples(path: Path) -> np.ndarray:
    """
    Load integer quadruples:
        h r o t

    Supports spaces, tabs, or commas.
    """

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            parts = re.split(r"[\s,]+", line)

            if len(parts) < 4:
                raise ValueError(
                    f"{path}:{line_no}: expected at least 4 columns, got {parts}"
                )

            try:
                h, r, o, t = map(int, parts[:4])
            except ValueError as e:
                raise ValueError(
                    f"{path}:{line_no}: first 4 columns must be integer IDs. "
                    f"Got: {parts[:4]}"
                ) from e

            rows.append((h, r, o, t))

    return np.asarray(rows, dtype=np.int64)


def load_time2id(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    return {str(k): int(v) for k, v in obj.items()}


def unique_time_targets(quads: np.ndarray) -> np.ndarray:
    """
    Keep only triples (h,r,o) having exactly ONE distinct timestamp
    inside the evaluated split.

    Returns one row per unique-time triple:
        h r o t
    """

    triple_to_times = defaultdict(set)

    for h, r, o, t in quads:
        key = (int(h), int(r), int(o))
        triple_to_times[key].add(int(t))

    targets = []

    for (h, r, o), times in triple_to_times.items():
        if len(times) == 1:
            t = next(iter(times))
            targets.append((h, r, o, t))

    return np.asarray(targets, dtype=np.int64)


def sample_targets(
    targets: np.ndarray,
    max_targets: Optional[int],
    seed: int
) -> np.ndarray:

    if max_targets is None or max_targets <= 0:
        return targets

    if len(targets) <= max_targets:
        return targets

    rng = np.random.default_rng(seed)

    idx = rng.choice(
        len(targets),
        size=max_targets,
        replace=False
    )

    return targets[idx]


# ============================================================
# Timestamp baselines
# ============================================================

def timestamp_frequency_ranking(
    train_quads: np.ndarray,
    all_time_ids: np.ndarray
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Build frequency baseline from victim_train ONLY.

    Most frequent timestamp gets rank 1.
    """

    counter = Counter(int(t) for t in train_quads[:, 3])

    ordered = sorted(
        (int(t) for t in all_time_ids),
        key=lambda t: (-counter.get(t, 0), t)
    )

    rank_map = {
        t: i + 1
        for i, t in enumerate(ordered)
    }

    return np.asarray(ordered, dtype=np.int64), rank_map


def chronological_positions(
    time2id: Dict[str, int]
) -> Dict[int, int]:
    """
    Map time IDs to chronological positions.

    Supports:
    - YYYY-MM-DD
    - integer years
    - fallback lexical sorting
    """

    pairs = []

    for ts, tid in time2id.items():

        m = re.fullmatch(
            r"(\d{4})-(\d{1,2})-(\d{1,2})",
            ts
        )

        if m:
            y, mo, d = map(int, m.groups())
            sort_key = (0, y, mo, d)

        elif re.fullmatch(r"-?\d{1,6}", ts):
            sort_key = (1, int(ts), 0, 0)

        else:
            sort_key = (2, ts)

        pairs.append((sort_key, int(tid)))

    pairs.sort(key=lambda x: x[0])

    return {
        tid: pos
        for pos, (_, tid) in enumerate(pairs)
    }


# ============================================================
# Checkpoint utilities
# ============================================================

def unwrap_state_dict(ckpt):
    """
    Accept:
    - raw state_dict
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - {"model": ...}
    """

    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported checkpoint object.")

    state = None

    for key in [
        "model_state_dict",
        "state_dict",
        "model",
    ]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break

    if state is None:
        # Maybe the checkpoint itself is the raw state_dict
        if any(torch.is_tensor(v) for v in ckpt.values()):
            state = ckpt
        else:
            raise ValueError(
                "Could not find a valid model state_dict in checkpoint."
            )

    clean = {}

    for key, value in state.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        clean[new_key] = value

    return clean


def get_weight(
    state: Dict[str, torch.Tensor],
    aliases: List[str],
    description: str
) -> torch.Tensor:
    """
    Search embedding tensor by exact alias first,
    then by suffix match.
    """

    for alias in aliases:
        if alias in state:
            return state[alias].detach()

    for alias in aliases:
        matches = [
            key
            for key in state.keys()
            if key.endswith(alias)
        ]

        if len(matches) == 1:
            return state[matches[0]].detach()

    available = "\n  - ".join(sorted(state.keys()))

    raise KeyError(
        f"\nCannot find {description}.\n"
        f"Tried aliases:\n  - "
        + "\n  - ".join(aliases)
        + "\nAvailable checkpoint keys:\n  - "
        + available
    )


# ============================================================
# Model scorers
# ============================================================

class BaseScorer:

    def __init__(self, device):
        self.device = device

    @torch.inference_mode()
    def score_grid(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        o: torch.Tensor,
        time_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        h,r,o:      [B]
        time_ids:   [C]

        Return:
            [B,C]
        """
        raise NotImplementedError


# ------------------------------------------------------------
# TeRo-like
# ------------------------------------------------------------

class TeRoScorer(BaseScorer):

    def __init__(self, state, device):
        super().__init__(device)

        self.ent_re = get_weight(
            state,
            [
                "ent_re.weight",
                "entity_re.weight",
                "entities_re.weight",
            ],
            "TeRo entity real embedding"
        ).to(device)

        self.ent_im = get_weight(
            state,
            [
                "ent_im.weight",
                "entity_im.weight",
                "entities_im.weight",
            ],
            "TeRo entity imaginary embedding"
        ).to(device)

        self.rel_re = get_weight(
            state,
            [
                "rel_re.weight",
                "relation_re.weight",
                "relations_re.weight",
            ],
            "TeRo relation real embedding"
        ).to(device)

        self.rel_im = get_weight(
            state,
            [
                "rel_im.weight",
                "relation_im.weight",
                "relations_im.weight",
            ],
            "TeRo relation imaginary embedding"
        ).to(device)

        self.phase = get_weight(
            state,
            [
                "time_emb.weight",
                "time.weight",
                "temporal.weight",
                "time_embedding.weight",
            ],
            "TeRo temporal phase embedding"
        ).to(device)

    @torch.inference_mode()
    def score_grid(
        self,
        h,
        r,
        o,
        time_ids
    ):

        # [B,D]
        h_re = self.ent_re[h]
        h_im = self.ent_im[h]

        o_re = self.ent_re[o]
        o_im = self.ent_im[o]

        r_re = self.rel_re[r]
        r_im = self.rel_im[r]

        # [C,D]
        phase = self.phase[time_ids]

        cos_t = torch.cos(phase)
        sin_t = torch.sin(phase)

        # [B,C,D]
        h_t_re = (
            h_re[:, None, :] * cos_t[None, :, :]
            -
            h_im[:, None, :] * sin_t[None, :, :]
        )

        h_t_im = (
            h_re[:, None, :] * sin_t[None, :, :]
            +
            h_im[:, None, :] * cos_t[None, :, :]
        )

        o_t_re = (
            o_re[:, None, :] * cos_t[None, :, :]
            -
            o_im[:, None, :] * sin_t[None, :, :]
        )

        o_t_im = (
            o_re[:, None, :] * sin_t[None, :, :]
            +
            o_im[:, None, :] * cos_t[None, :, :]
        )

        # Relation is added AFTER temporal rotation
        diff_re = (
            h_t_re
            + r_re[:, None, :]
            - o_t_re
        )

        diff_im = (
            h_t_im
            + r_im[:, None, :]
            - o_t_im
        )

        distance = torch.sqrt(
            torch.sum(
                diff_re * diff_re
                +
                diff_im * diff_im,
                dim=-1
            )
            + 1e-12
        )

        return -distance


# ------------------------------------------------------------
# TComplEx
# ------------------------------------------------------------

class TComplExScorer(BaseScorer):

    def __init__(self, state, device):
        super().__init__(device)

        self.ent_re = get_weight(
            state,
            [
                "ent_re.weight",
                "entity_re.weight",
                "entities_re.weight",
            ],
            "TComplEx entity real embedding"
        ).to(device)

        self.ent_im = get_weight(
            state,
            [
                "ent_im.weight",
                "entity_im.weight",
                "entities_im.weight",
            ],
            "TComplEx entity imaginary embedding"
        ).to(device)

        self.rel_re = get_weight(
            state,
            [
                "rel_re.weight",
                "relation_re.weight",
                "relations_re.weight",
            ],
            "TComplEx relation real embedding"
        ).to(device)

        self.rel_im = get_weight(
            state,
            [
                "rel_im.weight",
                "relation_im.weight",
                "relations_im.weight",
            ],
            "TComplEx relation imaginary embedding"
        ).to(device)

        self.time_re = get_weight(
            state,
            [
                "time_re.weight",
                "temporal_re.weight",
                "times_re.weight",
            ],
            "TComplEx time real embedding"
        ).to(device)

        self.time_im = get_weight(
            state,
            [
                "time_im.weight",
                "temporal_im.weight",
                "times_im.weight",
            ],
            "TComplEx time imaginary embedding"
        ).to(device)

    @torch.inference_mode()
    def score_grid(
        self,
        h,
        r,
        o,
        time_ids
    ):

        # h * r
        hr_re = (
            self.ent_re[h] * self.rel_re[r]
            -
            self.ent_im[h] * self.rel_im[r]
        )

        hr_im = (
            self.ent_re[h] * self.rel_im[r]
            +
            self.ent_im[h] * self.rel_re[r]
        )

        # time
        t_re = self.time_re[time_ids]
        t_im = self.time_im[time_ids]

        # (h * r) * t
        q_re = (
            hr_re[:, None, :] * t_re[None, :, :]
            -
            hr_im[:, None, :] * t_im[None, :, :]
        )

        q_im = (
            hr_re[:, None, :] * t_im[None, :, :]
            +
            hr_im[:, None, :] * t_re[None, :, :]
        )

        # Re(q * conj(o))
        o_re = self.ent_re[o][:, None, :]
        o_im = self.ent_im[o][:, None, :]

        scores = torch.sum(
            q_re * o_re
            +
            q_im * o_im,
            dim=-1
        )

        return scores


# ------------------------------------------------------------
# TNTComplEx
# ------------------------------------------------------------

class TNTComplExScorer(BaseScorer):

    def __init__(self, state, device):
        super().__init__(device)

        self.ent_re = get_weight(
            state,
            [
                "ent_re.weight",
                "entity_re.weight",
                "entities_re.weight",
            ],
            "TNTComplEx entity real embedding"
        ).to(device)

        self.ent_im = get_weight(
            state,
            [
                "ent_im.weight",
                "entity_im.weight",
                "entities_im.weight",
            ],
            "TNTComplEx entity imaginary embedding"
        ).to(device)

        # IMPORTANT:
        # Current checkpoints use:
        #   rel_time_re.weight
        #   rel_time_im.weight

        self.rel_time_re = get_weight(
            state,
            [
                "rel_time_re.weight",
                "rel_temporal_re.weight",
                "rel_temp_re.weight",
                "relation_temporal_re.weight",
            ],
            "TNTComplEx temporal relation real embedding"
        ).to(device)

        self.rel_time_im = get_weight(
            state,
            [
                "rel_time_im.weight",
                "rel_temporal_im.weight",
                "rel_temp_im.weight",
                "relation_temporal_im.weight",
            ],
            "TNTComplEx temporal relation imaginary embedding"
        ).to(device)

        self.rel_static_re = get_weight(
            state,
            [
                "rel_static_re.weight",
                "relation_static_re.weight",
                "rel_non_temporal_re.weight",
            ],
            "TNTComplEx static relation real embedding"
        ).to(device)

        self.rel_static_im = get_weight(
            state,
            [
                "rel_static_im.weight",
                "relation_static_im.weight",
                "rel_non_temporal_im.weight",
            ],
            "TNTComplEx static relation imaginary embedding"
        ).to(device)

        self.time_re = get_weight(
            state,
            [
                "time_re.weight",
                "temporal_re.weight",
                "times_re.weight",
            ],
            "TNTComplEx time real embedding"
        ).to(device)

        self.time_im = get_weight(
            state,
            [
                "time_im.weight",
                "temporal_im.weight",
                "times_im.weight",
            ],
            "TNTComplEx time imaginary embedding"
        ).to(device)

    @torch.inference_mode()
    def score_grid(
        self,
        h,
        r,
        o,
        time_ids
    ):

        # Temporal relation component
        r_time_re = self.rel_time_re[r]
        r_time_im = self.rel_time_im[r]

        # Static relation component
        r_static_re = self.rel_static_re[r]
        r_static_im = self.rel_static_im[r]

        # Time embeddings
        t_re = self.time_re[time_ids]
        t_im = self.time_im[time_ids]

        # r_time * t
        rt_re = (
            r_time_re[:, None, :] * t_re[None, :, :]
            -
            r_time_im[:, None, :] * t_im[None, :, :]
        )

        rt_im = (
            r_time_re[:, None, :] * t_im[None, :, :]
            +
            r_time_im[:, None, :] * t_re[None, :, :]
        )

        # r_eff(t) = r_time * t + r_static
        r_eff_re = (
            rt_re
            +
            r_static_re[:, None, :]
        )

        r_eff_im = (
            rt_im
            +
            r_static_im[:, None, :]
        )

        # h * r_eff
        h_re = self.ent_re[h][:, None, :]
        h_im = self.ent_im[h][:, None, :]

        q_re = (
            h_re * r_eff_re
            -
            h_im * r_eff_im
        )

        q_im = (
            h_re * r_eff_im
            +
            h_im * r_eff_re
        )

        # Re(q * conj(o))
        o_re = self.ent_re[o][:, None, :]
        o_im = self.ent_im[o][:, None, :]

        scores = torch.sum(
            q_re * o_re
            +
            q_im * o_im,
            dim=-1
        )

        return scores


def build_scorer(
    model_name: str,
    checkpoint_path: Path,
    device
):

    # The checkpoints are our own trusted files.
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    state = unwrap_state_dict(ckpt)

    if model_name == "tero":
        return TeRoScorer(
            state,
            device
        )

    if model_name == "tcomplex":
        return TComplExScorer(
            state,
            device
        )

    if model_name == "tntcomplex":
        return TNTComplExScorer(
            state,
            device
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# Attack evaluation
# ============================================================

@torch.inference_mode()
def evaluate_model_attack(
    scorer: BaseScorer,
    targets: np.ndarray,
    all_time_ids: np.ndarray,
    chronological_pos: Dict[int, int],
    device,
    target_batch_size: int = 64,
    time_chunk_size: int = 256,
):
    """
    Global timestamp inference.

    For each target:
      1. compute score of true timestamp
      2. query all timestamps
      3. rank the true timestamp
      4. keep Top-10 predictions
    """

    if len(targets) == 0:
        raise ValueError(
            "No targets to evaluate."
        )

    all_time_ids_t = torch.as_tensor(
        all_time_ids,
        dtype=torch.long,
        device=device
    )

    ranks = []
    top1_hits = []
    top3_hits = []
    top10_hits = []
    top1_pred_ids = []
    temporal_step_errors = []

    for b0 in range(
        0,
        len(targets),
        target_batch_size
    ):

        batch_np = targets[
            b0:b0 + target_batch_size
        ]

        h = torch.as_tensor(
            batch_np[:, 0],
            dtype=torch.long,
            device=device
        )

        r = torch.as_tensor(
            batch_np[:, 1],
            dtype=torch.long,
            device=device
        )

        o = torch.as_tensor(
            batch_np[:, 2],
            dtype=torch.long,
            device=device
        )

        true_t = torch.as_tensor(
            batch_np[:, 3],
            dtype=torch.long,
            device=device
        )

        B = len(batch_np)

        # ----------------------------------------------------
        # True timestamp score for each target
        # ----------------------------------------------------

        true_scores = torch.empty(
            B,
            dtype=torch.float32,
            device=device
        )

        unique_true_times = torch.unique(
            true_t
        )

        for ut in unique_true_times:

            mask = (
                true_t == ut
            )

            idx = torch.where(
                mask
            )[0]

            scores = scorer.score_grid(
                h[idx],
                r[idx],
                o[idx],
                ut.view(1)
            ).squeeze(1)

            true_scores[idx] = scores.float()

        # Count scores greater than true score.
        better_counts = torch.zeros(
            B,
            dtype=torch.long,
            device=device
        )

        keep_k = min(
            10,
            len(all_time_ids)
        )

        best_scores = torch.full(
            (B, keep_k),
            -float("inf"),
            dtype=torch.float32,
            device=device
        )

        best_tids = torch.full(
            (B, keep_k),
            -1,
            dtype=torch.long,
            device=device
        )

        # ----------------------------------------------------
        # Query ALL timestamps in chunks
        # ----------------------------------------------------

        for c0 in range(
            0,
            len(all_time_ids),
            time_chunk_size
        ):

            tids = all_time_ids_t[
                c0:c0 + time_chunk_size
            ]

            scores = scorer.score_grid(
                h,
                r,
                o,
                tids
            ).float()

            # Rank = 1 + number of timestamps with higher score
            better_counts += torch.sum(
                scores > true_scores[:, None],
                dim=1
            )

            # Merge current top-K
            candidate_scores = torch.cat(
                [
                    best_scores,
                    scores,
                ],
                dim=1
            )

            candidate_tids = torch.cat(
                [
                    best_tids,
                    tids[None, :].expand(
                        B,
                        -1
                    ),
                ],
                dim=1
            )

            vals, idxs = torch.topk(
                candidate_scores,
                k=keep_k,
                dim=1
            )

            best_scores = vals

            best_tids = torch.gather(
                candidate_tids,
                1,
                idxs
            )

        batch_ranks = (
            better_counts + 1
        ).detach().cpu().numpy()

        true_np = batch_np[
            :, 3
        ].astype(
            np.int64
        )

        top_ids_np = (
            best_tids
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Metrics per target
        # ----------------------------------------------------

        for i in range(B):

            rank = int(
                batch_ranks[i]
            )

            true_id = int(
                true_np[i]
            )

            predicted_id = int(
                top_ids_np[i, 0]
            )

            ranks.append(
                rank
            )

            top1_hits.append(
                int(
                    true_id in top_ids_np[
                        i,
                        :min(1, keep_k)
                    ]
                )
            )

            top3_hits.append(
                int(
                    true_id in top_ids_np[
                        i,
                        :min(3, keep_k)
                    ]
                )
            )

            top10_hits.append(
                int(
                    true_id in top_ids_np[
                        i,
                        :min(10, keep_k)
                    ]
                )
            )

            top1_pred_ids.append(
                predicted_id
            )

            if (
                true_id in chronological_pos
                and
                predicted_id in chronological_pos
            ):

                temporal_step_errors.append(
                    abs(
                        chronological_pos[predicted_id]
                        -
                        chronological_pos[true_id]
                    )
                )

    ranks = np.asarray(
        ranks,
        dtype=np.float64
    )

    result = {
        "n_targets": int(
            len(targets)
        ),

        "top1": float(
            np.mean(top1_hits)
        ),

        "top3": float(
            np.mean(top3_hits)
        ),

        "top10": float(
            np.mean(top10_hits)
        ),

        "mrr": float(
            np.mean(
                1.0 / ranks
            )
        ),

        "mean_rank": float(
            np.mean(ranks)
        ),

        "median_rank": float(
            np.median(ranks)
        ),

        "mean_temporal_step_error": (
            float(
                np.mean(
                    temporal_step_errors
                )
            )
            if temporal_step_errors
            else float("nan")
        ),

        "median_temporal_step_error": (
            float(
                np.median(
                    temporal_step_errors
                )
            )
            if temporal_step_errors
            else float("nan")
        ),
    }

    return (
        result,
        ranks,
        np.asarray(
            top1_pred_ids,
            dtype=np.int64
        )
    )


# ============================================================
# Baseline evaluation
# ============================================================

def evaluate_frequency_baseline(
    targets: np.ndarray,
    freq_order: np.ndarray,
    freq_rank_map: Dict[int, int],
    chronological_pos: Dict[int, int]
):

    ranks = np.asarray(
        [
            freq_rank_map[int(t)]
            for t in targets[:, 3]
        ],
        dtype=np.float64
    )

    top1_pred = int(
        freq_order[0]
    )

    temporal_errors = []

    for t in targets[:, 3]:

        true_t = int(t)

        if (
            true_t in chronological_pos
            and
            top1_pred in chronological_pos
        ):

            temporal_errors.append(
                abs(
                    chronological_pos[top1_pred]
                    -
                    chronological_pos[true_t]
                )
            )

    return {
        "n_targets": int(
            len(targets)
        ),

        "top1": float(
            np.mean(
                ranks <= 1
            )
        ),

        "top3": float(
            np.mean(
                ranks <= 3
            )
        ),

        "top10": float(
            np.mean(
                ranks <= 10
            )
        ),

        "mrr": float(
            np.mean(
                1.0 / ranks
            )
        ),

        "mean_rank": float(
            np.mean(ranks)
        ),

        "median_rank": float(
            np.median(ranks)
        ),

        "mean_temporal_step_error": (
            float(
                np.mean(
                    temporal_errors
                )
            )
            if temporal_errors
            else float("nan")
        ),

        "median_temporal_step_error": (
            float(
                np.median(
                    temporal_errors
                )
            )
            if temporal_errors
            else float("nan")
        ),
    }


def random_baseline(
    num_times: int
):

    harmonic = sum(
        1.0 / i
        for i in range(
            1,
            num_times + 1
        )
    )

    return {
        "n_targets": None,

        "top1": min(
            1.0,
            1.0 / num_times
        ),

        "top3": min(
            1.0,
            3.0 / num_times
        ),

        "top10": min(
            1.0,
            10.0 / num_times
        ),

        "mrr": (
            harmonic / num_times
        ),

        "mean_rank": (
            num_times + 1
        ) / 2.0,

        "median_rank": float(
            math.ceil(
                num_times / 2
            )
        ),

        "mean_temporal_step_error": float(
            "nan"
        ),

        "median_temporal_step_error": float(
            "nan"
        ),
    }


# ============================================================
# Output details
# ============================================================

def write_target_details(
    path: Path,
    targets: np.ndarray,
    ranks: np.ndarray,
    top1_pred_ids: np.ndarray,
    chronological_pos: Dict[int, int],
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "h",
            "r",
            "o",
            "true_t",
            "pred_t_top1",
            "rank_true_t",
            "reciprocal_rank",
            "temporal_step_error",
        ])

        for (
            row,
            rank,
            pred
        ) in zip(
            targets,
            ranks,
            top1_pred_ids
        ):

            h, r, o, true_t = map(
                int,
                row
            )

            pred = int(
                pred
            )

            temporal_error = ""

            if (
                true_t in chronological_pos
                and
                pred in chronological_pos
            ):

                temporal_error = abs(
                    chronological_pos[pred]
                    -
                    chronological_pos[true_t]
                )

            writer.writerow([
                h,
                r,
                o,
                true_t,
                pred,
                int(rank),
                1.0 / float(rank),
                temporal_error,
            ])


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--max-targets",
        type=int,
        default=5000,
        help=(
            "Maximum number of targets per split. "
            "Use 0 to evaluate ALL unique-time triples."
        )
    )

    parser.add_argument(
        "--target-batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--time-chunk-size",
        type=int,
        default=256
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    root = Path(
        args.root
    )

    processed_root = (
        root
        /
        "processed"
    )

    models_root = (
        root
        /
        "models"
    )

    results_root = (
        root
        /
        "results"
        /
        "timestamp_inference"
    )

    results_root.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        args.device
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Root:   {root}"
    )

    summary_rows = []

    # ========================================================
    # Dataset loop
    # ========================================================

    for dataset in args.datasets:

        print(
            "\n"
            +
            "=" * 88
        )

        print(
            f"DATASET: {dataset}"
        )

        print(
            "=" * 88
        )

        dataset_dir = (
            processed_root
            /
            dataset
        )

        train_path = (
            dataset_dir
            /
            "victim_train.txt"
        )

        test_path = (
            dataset_dir
            /
            "victim_test.txt"
        )

        time_path = (
            dataset_dir
            /
            "time2id.json"
        )

        for path in [
            train_path,
            test_path,
            time_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    path
                )

        train_quads = load_quadruples(
            train_path
        )

        test_quads = load_quadruples(
            test_path
        )

        time2id = load_time2id(
            time_path
        )

        all_time_ids = np.asarray(
            sorted(
                set(
                    time2id.values()
                )
            ),
            dtype=np.int64
        )

        chronological_pos = chronological_positions(
            time2id
        )

        train_targets_all = unique_time_targets(
            train_quads
        )

        test_targets_all = unique_time_targets(
            test_quads
        )

        max_targets = (
            None
            if args.max_targets <= 0
            else args.max_targets
        )

        train_targets = sample_targets(
            train_targets_all,
            max_targets,
            args.seed
        )

        test_targets = sample_targets(
            test_targets_all,
            max_targets,
            args.seed + 1
        )

        print(
            f"timestamps: {len(all_time_ids):,}"
        )

        print(
            f"unique-time train triples: "
            f"{len(train_targets_all):,}; "
            f"evaluated: {len(train_targets):,}"
        )

        print(
            f"unique-time test triples : "
            f"{len(test_targets_all):,}; "
            f"evaluated: {len(test_targets):,}"
        )

        # ====================================================
        # Baselines
        # ====================================================

        freq_order, freq_rank_map = timestamp_frequency_ranking(
            train_quads,
            all_time_ids
        )

        random_result = random_baseline(
            len(all_time_ids)
        )

        for split_name, targets in [
            ("train", train_targets),
            ("test", test_targets),
        ]:

            if len(targets) == 0:
                continue

            frequency_result = evaluate_frequency_baseline(
                targets,
                freq_order,
                freq_rank_map,
                chronological_pos
            )

            for baseline_name, result in [
                ("random", random_result),
                ("frequency", frequency_result),
            ]:

                summary_rows.append({
                    "dataset": dataset,
                    "model": baseline_name,
                    "split": split_name,
                    "num_timestamps": len(all_time_ids),
                    "n_targets": (
                        len(targets)
                        if result["n_targets"] is None
                        else result["n_targets"]
                    ),
                    "top1": result["top1"],
                    "top3": result["top3"],
                    "top10": result["top10"],
                    "mrr": result["mrr"],
                    "mean_rank": result["mean_rank"],
                    "median_rank": result["median_rank"],
                    "mean_temporal_step_error": (
                        result["mean_temporal_step_error"]
                    ),
                    "median_temporal_step_error": (
                        result["median_temporal_step_error"]
                    ),
                })

        # ====================================================
        # Model loop
        # ====================================================

        for model_name in args.models:

            checkpoint_path = (
                models_root
                /
                dataset
                /
                f"{model_name}_victim.pt"
            )

            if not checkpoint_path.exists():

                print(
                    f"[WARN] Missing checkpoint, skipping: "
                    f"{checkpoint_path}"
                )

                continue

            print(
                f"\n--- {model_name.upper()} ---"
            )

            scorer = build_scorer(
                model_name,
                checkpoint_path,
                device
            )

            for split_name, targets in [
                ("train", train_targets),
                ("test", test_targets),
            ]:

                if len(targets) == 0:
                    continue

                print(
                    f"Evaluating {split_name}: "
                    f"{len(targets):,} targets x "
                    f"{len(all_time_ids):,} timestamps"
                )

                result, ranks, predictions = evaluate_model_attack(
                    scorer=scorer,
                    targets=targets,
                    all_time_ids=all_time_ids,
                    chronological_pos=chronological_pos,
                    device=device,
                    target_batch_size=args.target_batch_size,
                    time_chunk_size=args.time_chunk_size,
                )

                print(
                    f"{split_name:5s} | "
                    f"Top1={result['top1']:.6f} "
                    f"Top3={result['top3']:.6f} "
                    f"Top10={result['top10']:.6f} "
                    f"MRR={result['mrr']:.6f} "
                    f"MedRank={result['median_rank']:.1f} "
                    f"MedTimeErr="
                    f"{result['median_temporal_step_error']:.1f}"
                )

                summary_rows.append({
                    "dataset": dataset,
                    "model": model_name,
                    "split": split_name,
                    "num_timestamps": len(all_time_ids),
                    **result,
                })

                details_path = (
                    results_root
                    /
                    dataset
                    /
                    f"{model_name}_{split_name}_details.csv"
                )

                write_target_details(
                    details_path,
                    targets,
                    ranks,
                    predictions,
                    chronological_pos
                )

            del scorer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ========================================================
    # Save global summary
    # ========================================================

    summary_path = (
        results_root
        /
        "timestamp_inference_summary.csv"
    )

    fieldnames = [
        "dataset",
        "model",
        "split",
        "num_timestamps",
        "n_targets",
        "top1",
        "top3",
        "top10",
        "mrr",
        "mean_rank",
        "median_rank",
        "mean_temporal_step_error",
        "median_temporal_step_error",
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in summary_rows:
            writer.writerow(
                row
            )

    print(
        "\n"
        +
        "=" * 88
    )

    print(
        "DONE"
    )

    print(
        f"Summary saved to:\n{summary_path}"
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()
