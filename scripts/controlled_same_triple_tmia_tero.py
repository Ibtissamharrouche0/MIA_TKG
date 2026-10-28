#!/usr/bin/env python3
# ============================================================
# controlled_same_triple_tmia_tero.py
#
# Same-triple controlled MIA / T-MIA for the TeRo-like model.
#
# IMPORTANT:
#   This script REUSES the exact same (h,r,o,t_member,t_nonmember)
#   pairs previously generated for the TComplEx controlled test.
#   Therefore the TeRo-like vs TComplEx comparison is aligned:
#   same datasets, same candidate pairs, same temporal window,
#   same shadow->victim LR attacker.
#
# Pair source:
#   attack_data_tcomplex_same_triple/<DATASET>/
#       shadow_same_triple_pairs.csv
#       victim_same_triple_pairs.csv
#
# TeRo-like checkpoints:
#   models/<DATASET>/tero_shadow.pt
#   models/<DATASET>/tero_victim.pt
#
# Outputs:
#   attack_data_tero_same_triple/<DATASET>/
#       shadow_same_triple_attack.csv
#       victim_same_triple_attack.csv
#
# Final results:
#   results/tero_same_triple_tmia_results.csv
# ============================================================

from pathlib import Path
from datetime import datetime, timedelta
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("/home/harrouch/Paper2/Paper2")

PROCESSED_ROOT = BASE_DIR / "processed"
MODEL_ROOT = BASE_DIR / "models"

# Reuse EXACT TComplEx-controlled pair files
PAIR_ROOT = BASE_DIR / "attack_data_tcomplex_same_triple"

# TeRo controlled attack features
OUTPUT_ROOT = BASE_DIR / "attack_data_tero_same_triple"

RESULTS_DIR = BASE_DIR / "results"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "ICEWS05-15",
    "ICEWS14",
    "YAGO11k",
    "WIKIDATA12k",
]

ICEWS_DATASETS = {
    "ICEWS05-15",
    "ICEWS14",
}

YEAR_DATASETS = {
    "YAGO11k",
    "WIKIDATA12k",
}

SEED = 42
BATCH_SIZE = 4096

ATTACK_OFFSETS = [
    -2,
    -1,
    0,
    1,
    2,
]

BASELINE_FEATURES = [
    "score_t",
]

TEMPORAL_FEATURES = [
    "score_t_minus_2",
    "score_t_minus_1",
    "score_t",
    "score_t_plus_1",
    "score_t_plus_2",
]

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# TeRo-like MODEL
# ============================================================

class TeRoLike(nn.Module):
    """
    Scoring is aligned with the corrected TeRo-like attack generator:

      phase = time_emb(t)

      h_t = rotate(h, phase)
      o_t = rotate(o, phase)

      pred = h_t + r

      score = - || pred - o_t ||_2

    Relation is added AFTER temporal rotation.
    """

    def __init__(
        self,
        num_entities,
        num_relations,
        num_times,
        embedding_dim,
    ):
        super().__init__()

        self.ent_re = nn.Embedding(
            num_entities,
            embedding_dim,
        )

        self.ent_im = nn.Embedding(
            num_entities,
            embedding_dim,
        )

        self.rel_re = nn.Embedding(
            num_relations,
            embedding_dim,
        )

        self.rel_im = nn.Embedding(
            num_relations,
            embedding_dim,
        )

        self.time_emb = nn.Embedding(
            num_times,
            embedding_dim,
        )

    @staticmethod
    def rotate(
        re,
        im,
        phase,
    ):
        cos_phase = torch.cos(
            phase
        )

        sin_phase = torch.sin(
            phase
        )

        rotated_re = (
            re * cos_phase
            -
            im * sin_phase
        )

        rotated_im = (
            re * sin_phase
            +
            im * cos_phase
        )

        return (
            rotated_re,
            rotated_im,
        )

    def score(
        self,
        h,
        r,
        o,
        t,
    ):
        h_re = self.ent_re(
            h
        )

        h_im = self.ent_im(
            h
        )

        o_re = self.ent_re(
            o
        )

        o_im = self.ent_im(
            o
        )

        r_re = self.rel_re(
            r
        )

        r_im = self.rel_im(
            r
        )

        phase = self.time_emb(
            t
        )

        h_t_re, h_t_im = (
            self.rotate(
                h_re,
                h_im,
                phase,
            )
        )

        o_t_re, o_t_im = (
            self.rotate(
                o_re,
                o_im,
                phase,
            )
        )

        pred_re = (
            h_t_re
            +
            r_re
        )

        pred_im = (
            h_t_im
            +
            r_im
        )

        diff_re = (
            pred_re
            -
            o_t_re
        )

        diff_im = (
            pred_im
            -
            o_t_im
        )

        distance = torch.sqrt(
            torch.sum(
                diff_re.pow(2)
                +
                diff_im.pow(2),
                dim=1,
            )
            +
            1e-12
        )

        return -distance


# ============================================================
# TIME MAPPING
# ============================================================

def load_time_mapping(
    dataset_name,
):
    path = (
        PROCESSED_ROOT
        /
        dataset_name
        /
        "time2id.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        raw = json.load(
            f
        )

    time2id = {
        str(k): int(v)
        for k, v
        in raw.items()
    }

    id2time = {
        int(v): str(k)
        for k, v
        in raw.items()
    }

    return (
        time2id,
        id2time,
    )


def get_neighbor_time_id(
    dataset_name,
    original_time_id,
    delta,
    time2id,
    id2time,
):
    """
    Same temporal-neighbor rule as the corrected TeRo/T-MIA pipeline:
      ICEWS       -> exact calendar day +/- delta
      YAGO/WIKI   -> exact year +/- delta
      missing time -> None / NaN
    """

    original_value = id2time[
        int(
            original_time_id
        )
    ]

    if dataset_name in ICEWS_DATASETS:

        current_date = datetime.strptime(
            original_value,
            "%Y-%m-%d",
        )

        neighbor_date = (
            current_date
            +
            timedelta(
                days=delta
            )
        )

        neighbor_value = (
            neighbor_date.strftime(
                "%Y-%m-%d"
            )
        )

    elif dataset_name in YEAR_DATASETS:

        current_year = int(
            original_value
        )

        neighbor_value = str(
            current_year
            +
            delta
        )

    else:
        raise ValueError(
            f"Unsupported dataset: "
            f"{dataset_name}"
        )

    return time2id.get(
        neighbor_value
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def clean_state_dict(
    state_dict,
):
    cleaned = {}

    for key, value in (
        state_dict.items()
    ):

        while key.startswith(
            "_module."
        ):
            key = key[
                len("_module.") :
            ]

        while key.startswith(
            "module."
        ):
            key = key[
                len("module.") :
            ]

        cleaned[
            key
        ] = value

    return cleaned


def extract_state_dict(
    checkpoint,
):
    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in checkpoint
    ):
        return checkpoint[
            "model_state_dict"
        ]

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "state_dict"
        in checkpoint
    ):
        return checkpoint[
            "state_dict"
        ]

    # Some simple checkpoints may directly be a state_dict.
    if isinstance(
        checkpoint,
        dict,
    ):
        required = {
            "ent_re.weight",
            "ent_im.weight",
            "rel_re.weight",
            "rel_im.weight",
            "time_emb.weight",
        }

        if required.issubset(
            set(
                checkpoint.keys()
            )
        ):
            return checkpoint

    raise ValueError(
        "Unsupported TeRo-like "
        "checkpoint structure."
    )


def load_tero_model(
    dataset_name,
    role,
    device,
):
    checkpoint_path = (
        MODEL_ROOT
        /
        dataset_name
        /
        f"tero_{role}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = clean_state_dict(
        extract_state_dict(
            checkpoint
        )
    )

    required_keys = [
        "ent_re.weight",
        "ent_im.weight",
        "rel_re.weight",
        "rel_im.weight",
        "time_emb.weight",
    ]

    for key in required_keys:
        if key not in state_dict:
            raise KeyError(
                f"Missing key "
                f"{key} in "
                f"{checkpoint_path}"
            )

    num_entities = (
        state_dict[
            "ent_re.weight"
        ].shape[0]
    )

    embedding_dim = (
        state_dict[
            "ent_re.weight"
        ].shape[1]
    )

    num_relations = (
        state_dict[
            "rel_re.weight"
        ].shape[0]
    )

    num_times = (
        state_dict[
            "time_emb.weight"
        ].shape[0]
    )

    model = TeRoLike(
        num_entities=
            num_entities,
        num_relations=
            num_relations,
        num_times=
            num_times,
        embedding_dim=
            embedding_dim,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    print(
        f"Loaded: "
        f"{checkpoint_path}"
    )

    print(
        f"Dimensions: "
        f"entities={num_entities}, "
        f"relations={num_relations}, "
        f"times={num_times}, "
        f"dim={embedding_dim}"
    )

    return model


# ============================================================
# REUSE SAME TCOMPLEX PAIRS
# ============================================================

def load_same_triple_pairs(
    dataset_name,
    role,
):
    pair_path = (
        PAIR_ROOT
        /
        dataset_name
        /
        f"{role}_same_triple_pairs.csv"
    )

    if not pair_path.exists():
        raise FileNotFoundError(
            f"Missing controlled pair file: "
            f"{pair_path}"
        )

    df = pd.read_csv(
        pair_path
    )

    required = {
        "h",
        "r",
        "o",
        "member_t",
        "nonmember_t",
    }

    missing = (
        required
        -
        set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"{pair_path} missing columns: "
            f"{sorted(missing)}"
        )

    # Convert explicitly to integer.
    for col in [
        "h",
        "r",
        "o",
        "member_t",
        "nonmember_t",
    ]:
        df[col] = (
            df[col]
            .astype(
                int
            )
        )

    members = []

    nonmembers = []

    pair_ids = []

    for pair_id, row in (
        df.reset_index(
            drop=True
        )
        .iterrows()
    ):

        h = int(
            row["h"]
        )

        r = int(
            row["r"]
        )

        o = int(
            row["o"]
        )

        member_t = int(
            row[
                "member_t"
            ]
        )

        nonmember_t = int(
            row[
                "nonmember_t"
            ]
        )

        members.append(
            (
                h,
                r,
                o,
                member_t,
            )
        )

        nonmembers.append(
            (
                h,
                r,
                o,
                nonmember_t,
            )
        )

        pair_ids.append(
            pair_id
        )

    # Strong sanity check:
    # every member/nonmember pair has exactly same h,r,o.
    for m, nm in zip(
        members,
        nonmembers,
    ):
        if m[:3] != nm[:3]:
            raise RuntimeError(
                "Pair file violates "
                "same-triple condition."
            )

        if m == nm:
            raise RuntimeError(
                "Member and non-member "
                "quadruple are identical."
            )

    print(
        f"Reused controlled pairs: "
        f"{len(members)}"
    )

    print(
        f"Source: {pair_path}"
    )

    return (
        members,
        nonmembers,
        pair_ids,
    )


# ============================================================
# BATCH SCORING
# ============================================================

def score_queries(
    model,
    queries,
    device,
):
    if len(
        queries
    ) == 0:
        return []

    scores = []

    with torch.no_grad():

        for start in range(
            0,
            len(queries),
            BATCH_SIZE,
        ):

            batch = queries[
                start :
                start + BATCH_SIZE
            ]

            tensor = torch.tensor(
                batch,
                dtype=torch.long,
                device=device,
            )

            batch_scores = model.score(
                tensor[:, 0],
                tensor[:, 1],
                tensor[:, 2],
                tensor[:, 3],
            )

            scores.extend(
                batch_scores
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

    return scores


def extract_temporal_features(
    dataset_name,
    model,
    facts,
    labels,
    pair_ids,
    time2id,
    id2time,
    device,
):
    rows = []

    for start in range(
        0,
        len(facts),
        BATCH_SIZE,
    ):

        batch = facts[
            start :
            start + BATCH_SIZE
        ]

        batch_labels = labels[
            start :
            start + BATCH_SIZE
        ]

        batch_pair_ids = pair_ids[
            start :
            start + BATCH_SIZE
        ]

        queries = []

        valid_flags = []

        for h, r, o, t in batch:

            flags = []

            for delta in (
                ATTACK_OFFSETS
            ):

                neighbor_t = (
                    get_neighbor_time_id(
                        dataset_name=
                            dataset_name,
                        original_time_id=
                            t,
                        delta=
                            delta,
                        time2id=
                            time2id,
                        id2time=
                            id2time,
                    )
                )

                if neighbor_t is None:

                    # Placeholder score;
                    # later converted to NaN.
                    queries.append(
                        (
                            h,
                            r,
                            o,
                            t,
                        )
                    )

                    flags.append(
                        False
                    )

                else:

                    queries.append(
                        (
                            h,
                            r,
                            o,
                            int(
                                neighbor_t
                            ),
                        )
                    )

                    flags.append(
                        True
                    )

            valid_flags.append(
                flags
            )

        all_scores = score_queries(
            model=
                model,
            queries=
                queries,
            device=
                device,
        )

        score_matrix = (
            np.asarray(
                all_scores,
                dtype=np.float32,
            )
            .reshape(
                len(
                    batch
                ),
                len(
                    ATTACK_OFFSETS
                ),
            )
        )

        for i, (
            h,
            r,
            o,
            t,
        ) in enumerate(
            batch
        ):

            values = (
                score_matrix[
                    i
                ]
                .astype(
                    float
                )
                .tolist()
            )

            for j in range(
                len(
                    values
                )
            ):
                if not (
                    valid_flags[
                        i
                    ][
                        j
                    ]
                ):
                    values[
                        j
                    ] = float(
                        "nan"
                    )

            rows.append(
                {
                    "pair_id":
                        batch_pair_ids[
                            i
                        ],

                    "h":
                        h,

                    "r":
                        r,

                    "o":
                        o,

                    "t":
                        t,

                    "score_t_minus_2":
                        values[0],

                    "score_t_minus_1":
                        values[1],

                    "score_t":
                        values[2],

                    "score_t_plus_1":
                        values[3],

                    "score_t_plus_2":
                        values[4],

                    "label":
                        batch_labels[
                            i
                        ],
                }
            )

    return rows


# ============================================================
# BUILD TeRo SAME-TRIPLE ATTACK DATA
# ============================================================

def build_role_dataset(
    dataset_name,
    role,
    device,
):
    print(
        "\n"
        +
        "=" * 100
    )

    print(
        f"{dataset_name} | "
        f"{role.upper()} | "
        f"TeRo SAME-TRIPLE"
    )

    print(
        "=" * 100
    )

    (
        members,
        nonmembers,
        pair_ids,
    ) = load_same_triple_pairs(
        dataset_name=
            dataset_name,
        role=
            role,
    )

    model = load_tero_model(
        dataset_name=
            dataset_name,
        role=
            role,
        device=
            device,
    )

    time2id, id2time = (
        load_time_mapping(
            dataset_name
        )
    )

    member_labels = [
        1
    ] * len(
        members
    )

    nonmember_labels = [
        0
    ] * len(
        nonmembers
    )

    print(
        "Scoring members..."
    )

    member_rows = (
        extract_temporal_features(
            dataset_name=
                dataset_name,
            model=
                model,
            facts=
                members,
            labels=
                member_labels,
            pair_ids=
                pair_ids,
            time2id=
                time2id,
            id2time=
                id2time,
            device=
                device,
        )
    )

    print(
        "Scoring same-triple "
        "non-members..."
    )

    nonmember_rows = (
        extract_temporal_features(
            dataset_name=
                dataset_name,
            model=
                model,
            facts=
                nonmembers,
            labels=
                nonmember_labels,
            pair_ids=
                pair_ids,
            time2id=
                time2id,
            id2time=
                id2time,
            device=
                device,
        )
    )

    df = pd.DataFrame(
        member_rows
        +
        nonmember_rows
    )

    # Deterministic shuffle only.
    df = df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(
        drop=True
    )

    output_dir = (
        OUTPUT_ROOT
        /
        dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        /
        f"{role}_same_triple_attack.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    clean = df.dropna(
        subset=
            TEMPORAL_FEATURES
    )

    print(
        f"All rows              : "
        f"{len(df)}"
    )

    print(
        f"Rows with full ±2     : "
        f"{len(clean)} "
        f"({100.0 * len(clean) / len(df):.2f}%)"
    )

    print(
        f"Saved: "
        f"{output_path}"
    )

    return df


# ============================================================
# ATTACK
# ============================================================

def run_attack(
    shadow_df,
    victim_df,
    features,
    attack_name,
):
    X_shadow = (
        shadow_df[
            features
        ]
        .to_numpy()
    )

    y_shadow = (
        shadow_df[
            "label"
        ]
        .to_numpy()
    )

    X_victim = (
        victim_df[
            features
        ]
        .to_numpy()
    )

    y_victim = (
        victim_df[
            "label"
        ]
        .to_numpy()
    )

    classifier = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    classifier.fit(
        X_shadow,
        y_shadow,
    )

    pred = classifier.predict(
        X_victim
    )

    prob = (
        classifier
        .predict_proba(
            X_victim
        )[:, 1]
    )

    auc = roc_auc_score(
        y_victim,
        prob,
    )

    return {
        "attack":
            attack_name,

        "accuracy":
            accuracy_score(
                y_victim,
                pred,
            ),

        "precision":
            precision_score(
                y_victim,
                pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_victim,
                pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_victim,
                pred,
                zero_division=0,
            ),

        "roc_auc":
            auc,

        "effective_auc":
            max(
                auc,
                1.0 - auc,
            ),

        "attack_advantage":
            abs(
                2.0 * auc
                -
                1.0
            ),

        "shadow_samples":
            len(
                shadow_df
            ),

        "victim_samples":
            len(
                victim_df
            ),
    }


def direct_score_auc(
    df,
):
    return roc_auc_score(
        df[
            "label"
        ],
        df[
            "score_t"
        ],
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 100
    )

    print(
        "TeRo-LIKE CONTROLLED "
        "SAME-TRIPLE MIA / T-MIA"
    )

    print(
        "=" * 100
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        print(
            "GPU   : "
            +
            torch.cuda.get_device_name(
                0
            )
        )

    all_results = []

    for dataset_name in DATASETS:

        # ----------------------------------------------------
        # Exact same controlled pair files as TComplEx
        # ----------------------------------------------------

        shadow_df = (
            build_role_dataset(
                dataset_name=
                    dataset_name,
                role=
                    "shadow",
                device=
                    device,
            )
        )

        victim_df = (
            build_role_dataset(
                dataset_name=
                    dataset_name,
                role=
                    "victim",
                device=
                    device,
            )
        )

        # ----------------------------------------------------
        # SAME full-window examples for Baseline and T-MIA
        # ----------------------------------------------------

        clean_shadow = (
            shadow_df
            .dropna(
                subset=
                    TEMPORAL_FEATURES
            )
            .reset_index(
                drop=True
            )
        )

        clean_victim = (
            victim_df
            .dropna(
                subset=
                    TEMPORAL_FEATURES
            )
            .reset_index(
                drop=True
            )
        )

        print(
            "\n"
            +
            "-" * 100
        )

        print(
            f"EVALUATION | "
            f"{dataset_name}"
        )

        print(
            "-" * 100
        )

        print(
            f"Clean shadow samples: "
            f"{len(clean_shadow)}"
        )

        print(
            f"Clean victim samples: "
            f"{len(clean_victim)}"
        )

        direct_auc = direct_score_auc(
            clean_victim
        )

        baseline = run_attack(
            shadow_df=
                clean_shadow,
            victim_df=
                clean_victim,
            features=
                BASELINE_FEATURES,
            attack_name=
                "Baseline-MIA",
        )

        temporal = run_attack(
            shadow_df=
                clean_shadow,
            victim_df=
                clean_victim,
            features=
                TEMPORAL_FEATURES,
            attack_name=
                "Temporal-MIA",
        )

        baseline[
            "dataset"
        ] = dataset_name

        temporal[
            "dataset"
        ] = dataset_name

        baseline[
            "direct_victim_score_auc"
        ] = direct_auc

        temporal[
            "direct_victim_score_auc"
        ] = direct_auc

        all_results.extend(
            [
                baseline,
                temporal,
            ]
        )

        print(
            f"Direct victim AUC(score_t): "
            f"{direct_auc:.6f}"
        )

        print(
            f"Baseline-MIA ROC-AUC       : "
            f"{baseline['roc_auc']:.6f}"
        )

        print(
            f"Temporal-MIA ROC-AUC       : "
            f"{temporal['roc_auc']:.6f}"
        )

        print(
            f"Temporal gain              : "
            f"{temporal['roc_auc'] - baseline['roc_auc']:+.6f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(
        all_results
    )

    columns = [
        "dataset",
        "attack",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "effective_auc",
        "attack_advantage",
        "direct_victim_score_auc",
        "shadow_samples",
        "victim_samples",
    ]

    results_df = (
        results_df[
            columns
        ]
    )

    output_path = (
        RESULTS_DIR
        /
        "tero_same_triple_tmia_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "FINAL SAME-TRIPLE "
        "TeRo MIA / T-MIA RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )
