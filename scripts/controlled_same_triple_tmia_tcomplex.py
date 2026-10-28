#!/usr/bin/env python3
# ============================================================
# controlled_same_triple_tmia_tcomplex.py
#
# Controlled temporal-membership diagnostic for TComplEx.
#
# Goal:
#   Remove the "seen (h,r,o) vs unseen (h,r,o)" shortcut.
#
# For every selected MEMBER:
#   (h,r,o,t_member) is an actual training quadruple.
#
# Construct a paired NON-MEMBER:
#   (h,r,o,t_nonmember)
# with the EXACT SAME (h,r,o), but a nearby real timestamp
# that exists in the dataset time vocabulary and is NOT a
# training quadruple for this role.
#
# Shadow:
#   attacker is trained from shadow model / shadow_train.
#
# Victim:
#   attacker is evaluated on victim model / victim_train.
#
# IMPORTANT:
#   This is an auxiliary controlled diagnostic.
#   It does NOT replace the main MIA protocol using
#   shadow_test / victim_test as natural non-members.
# ============================================================

from pathlib import Path
from datetime import datetime, timedelta
import json
import random
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
OUTPUT_ROOT = BASE_DIR / "attack_data_tcomplex_same_triple"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "ICEWS05-15",
    "ICEWS14",
    "YAGO11k",
    "WIKIDATA12k",
]

ROLES = [
    "shadow",
    "victim",
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

# First try closest timestamps, then slightly farther ones.
# This keeps the non-member temporally plausible while ensuring
# it is NOT in the training set.
NONMEMBER_OFFSETS = [
    -1, 1,
    -2, 2,
    -3, 3,
    -4, 4,
    -5, 5,
]

# T-MIA feature window used in the paper.
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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# TCOMPLEX MODEL
# ============================================================

class TComplEx(nn.Module):

    def __init__(
        self,
        num_entities,
        num_relations,
        num_times,
        embedding_dim,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.base_num_relations = num_relations
        self.total_num_relations = 2 * num_relations
        self.num_times = num_times
        self.embedding_dim = embedding_dim

        self.ent_re = nn.Embedding(
            num_entities,
            embedding_dim,
        )

        self.ent_im = nn.Embedding(
            num_entities,
            embedding_dim,
        )

        self.rel_re = nn.Embedding(
            self.total_num_relations,
            embedding_dim,
        )

        self.rel_im = nn.Embedding(
            self.total_num_relations,
            embedding_dim,
        )

        self.time_re = nn.Embedding(
            num_times,
            embedding_dim,
        )

        self.time_im = nn.Embedding(
            num_times,
            embedding_dim,
        )

    @staticmethod
    def complex_mul(
        a_re,
        a_im,
        b_re,
        b_im,
    ):
        out_re = (
            a_re * b_re
            -
            a_im * b_im
        )

        out_im = (
            a_re * b_im
            +
            a_im * b_re
        )

        return out_re, out_im

    def query_embedding(
        self,
        h,
        r,
        t,
    ):
        h_re = self.ent_re(h)
        h_im = self.ent_im(h)

        r_re = self.rel_re(r)
        r_im = self.rel_im(r)

        t_re = self.time_re(t)
        t_im = self.time_im(t)

        q_re, q_im = self.complex_mul(
            h_re,
            h_im,
            r_re,
            r_im,
        )

        q_re, q_im = self.complex_mul(
            q_re,
            q_im,
            t_re,
            t_im,
        )

        return q_re, q_im

    def score(
        self,
        h,
        r,
        o,
        t,
    ):
        q_re, q_im = self.query_embedding(
            h,
            r,
            t,
        )

        o_re = self.ent_re(o)
        o_im = self.ent_im(o)

        return torch.sum(
            q_re * o_re
            +
            q_im * o_im,
            dim=1,
        )


# ============================================================
# I/O
# ============================================================

def load_facts(path):
    facts = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 4:
                raise ValueError(
                    f"{path}, line {line_number}: "
                    f"expected 4 integer columns"
                )

            facts.append(
                tuple(
                    map(
                        int,
                        parts,
                    )
                )
            )

    return facts


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
        raw = json.load(f)

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


# ============================================================
# TEMPORAL NEIGHBOR
# ============================================================

def get_neighbor_time_id(
    dataset_name,
    original_time_id,
    delta,
    time2id,
    id2time,
):
    original_value = id2time[
        int(original_time_id)
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
            f"Unknown dataset: "
            f"{dataset_name}"
        )

    return time2id.get(
        neighbor_value
    )


# ============================================================
# MODEL LOADING
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

        cleaned[key] = value

    return cleaned


def load_model(
    dataset_name,
    role,
    device,
):
    checkpoint_path = (
        MODEL_ROOT
        /
        dataset_name
        /
        f"tcomplex_{role}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "state_dict"
        in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        raise ValueError(
            f"Unsupported checkpoint: "
            f"{checkpoint_path}"
        )

    state_dict = clean_state_dict(
        state_dict
    )

    num_entities = checkpoint.get(
        "num_entities",
        state_dict[
            "ent_re.weight"
        ].shape[0],
    )

    embedding_dim = checkpoint.get(
        "embedding_dim",
        state_dict[
            "ent_re.weight"
        ].shape[1],
    )

    num_times = checkpoint.get(
        "num_times",
        state_dict[
            "time_re.weight"
        ].shape[0],
    )

    num_relations = checkpoint.get(
        "num_relations",
        state_dict[
            "rel_re.weight"
        ].shape[0]
        //
        2,
    )

    model = TComplEx(
        num_entities=int(
            num_entities
        ),
        num_relations=int(
            num_relations
        ),
        num_times=int(
            num_times
        ),
        embedding_dim=int(
            embedding_dim
        ),
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    return model


# ============================================================
# SAME-TRIPLE PAIR CONSTRUCTION
# ============================================================

def construct_same_triple_pairs(
    dataset_name,
    train_facts,
    time2id,
    id2time,
    seed=42,
):
    """
    For a member (h,r,o,t), create a non-member
    (h,r,o,t') with exactly the same (h,r,o).

    t' must:
      - be a real timestamp in time2id
      - be close to t
      - NOT appear as (h,r,o,t') in train_facts

    We select at most one pair per member quadruple.
    """

    train_quad_set = set(
        train_facts
    )

    rng = random.Random(
        seed
    )

    shuffled_members = list(
        train_quad_set
    )

    rng.shuffle(
        shuffled_members
    )

    member_candidates = []
    nonmember_candidates = []
    pair_rows = []

    used_nonmember_quads = set()

    offset_hist = {}

    for h, r, o, t in (
        shuffled_members
    ):

        # Randomize the sign ordering a little while
        # staying reproducible.
        offsets = list(
            NONMEMBER_OFFSETS
        )

        # Small deterministic shuffle per member.
        rng.shuffle(
            offsets
        )

        chosen_t = None
        chosen_delta = None

        for delta in offsets:

            neighbor_t = (
                get_neighbor_time_id(
                    dataset_name=
                        dataset_name,
                    original_time_id=t,
                    delta=delta,
                    time2id=time2id,
                    id2time=id2time,
                )
            )

            if neighbor_t is None:
                continue

            candidate = (
                h,
                r,
                o,
                int(neighbor_t),
            )

            # Must be a true non-member relative to this model's train.
            if candidate in train_quad_set:
                continue

            # Avoid duplicate synthetic non-members.
            if candidate in used_nonmember_quads:
                continue

            chosen_t = int(
                neighbor_t
            )

            chosen_delta = int(
                delta
            )

            used_nonmember_quads.add(
                candidate
            )

            break

        if chosen_t is None:
            continue

        member = (
            h,
            r,
            o,
            t,
        )

        nonmember = (
            h,
            r,
            o,
            chosen_t,
        )

        member_candidates.append(
            member
        )

        nonmember_candidates.append(
            nonmember
        )

        pair_rows.append(
            {
                "h": h,
                "r": r,
                "o": o,
                "member_t": t,
                "nonmember_t":
                    chosen_t,
                "delta":
                    chosen_delta,
            }
        )

        offset_hist[
            chosen_delta
        ] = (
            offset_hist.get(
                chosen_delta,
                0,
            )
            +
            1
        )

    return (
        member_candidates,
        nonmember_candidates,
        pair_rows,
        offset_hist,
    )


# ============================================================
# SCORING
# ============================================================

def score_queries(
    model,
    queries,
    device,
):
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
                        original_time_id=t,
                        delta=delta,
                        time2id=time2id,
                        id2time=id2time,
                    )
                )

                if neighbor_t is not None:

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

                else:

                    # Placeholder; replaced by NaN.
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

            valid_flags.append(
                flags
            )

        all_scores = score_queries(
            model=model,
            queries=queries,
            device=device,
        )

        score_matrix = (
            np.asarray(
                all_scores,
                dtype=np.float32,
            )
            .reshape(
                len(batch),
                5,
            )
        )

        for i, (
            h,
            r,
            o,
            t,
        ) in enumerate(batch):

            values = (
                score_matrix[
                    i
                ]
                .astype(float)
                .tolist()
            )

            for j in range(5):

                if not valid_flags[i][j]:
                    values[j] = float(
                        "nan"
                    )

            rows.append(
                {
                    "pair_id":
                        batch_pair_ids[i],

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
                        batch_labels[i],
                }
            )

    return rows


# ============================================================
# BUILD CONTROLLED ATTACK DATASET
# ============================================================

def build_controlled_dataset(
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
        f"SAME-TRIPLE CONTROL"
    )

    print(
        "=" * 100
    )

    dataset_dir = (
        PROCESSED_ROOT
        /
        dataset_name
    )

    train_path = (
        dataset_dir
        /
        f"{role}_train.txt"
    )

    train_facts = load_facts(
        train_path
    )

    time2id, id2time = (
        load_time_mapping(
            dataset_name
        )
    )

    (
        members,
        nonmembers,
        pair_rows,
        offset_hist,
    ) = construct_same_triple_pairs(
        dataset_name=
            dataset_name,
        train_facts=
            train_facts,
        time2id=
            time2id,
        id2time=
            id2time,
        seed=SEED,
    )

    if len(members) == 0:
        raise RuntimeError(
            f"No controlled pairs "
            f"could be constructed for "
            f"{dataset_name} / {role}"
        )

    print(
        f"Training quadruples     : "
        f"{len(train_facts)}"
    )

    print(
        f"Controlled pairs        : "
        f"{len(members)}"
    )

    print(
        f"Controlled attack rows  : "
        f"{2 * len(members)}"
    )

    print(
        "Synthetic non-member "
        "offset histogram:"
    )

    for delta in sorted(
        offset_hist
    ):
        print(
            f"  delta {delta:+d}: "
            f"{offset_hist[delta]}"
        )

    # Verify strict same-triple condition.
    for member, nonmember in zip(
        members,
        nonmembers,
    ):
        assert (
            member[:3]
            ==
            nonmember[:3]
        )

        assert (
            member
            !=
            nonmember
        )

    train_set = set(
        train_facts
    )

    leaked_nonmembers = [
        q
        for q in nonmembers
        if q in train_set
    ]

    if leaked_nonmembers:
        raise RuntimeError(
            "Constructed non-members "
            "contain training members."
        )

    model = load_model(
        dataset_name=
            dataset_name,
        role=role,
        device=device,
    )

    # Same pair id for corresponding member/non-member.
    member_pair_ids = list(
        range(
            len(members)
        )
    )

    nonmember_pair_ids = list(
        range(
            len(nonmembers)
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
        "Scoring controlled members..."
    )

    member_rows = (
        extract_temporal_features(
            dataset_name=
                dataset_name,
            model=model,
            facts=members,
            labels=
                member_labels,
            pair_ids=
                member_pair_ids,
            time2id=
                time2id,
            id2time=
                id2time,
            device=device,
        )
    )

    print(
        "Scoring controlled non-members..."
    )

    nonmember_rows = (
        extract_temporal_features(
            dataset_name=
                dataset_name,
            model=model,
            facts=nonmembers,
            labels=
                nonmember_labels,
            pair_ids=
                nonmember_pair_ids,
            time2id=
                time2id,
            id2time=
                id2time,
            device=device,
        )
    )

    df = pd.DataFrame(
        member_rows
        +
        nonmember_rows
    )

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

    pair_df = pd.DataFrame(
        pair_rows
    )

    pair_path = (
        output_dir
        /
        f"{role}_same_triple_pairs.csv"
    )

    pair_df.to_csv(
        pair_path,
        index=False,
    )

    clean = df.dropna(
        subset=
            TEMPORAL_FEATURES
    )

    print(
        f"Rows with full ±2 window: "
        f"{len(clean)} / {len(df)} "
        f"({100.0 * len(clean) / len(df):.2f}%)"
    )

    print(
        f"Saved attack data: "
        f"{output_path}"
    )

    print(
        f"Saved pairs      : "
        f"{pair_path}"
    )

    return df


# ============================================================
# ATTACK EVALUATION
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
        "TCOMPLEX CONTROLLED "
        "SAME-TRIPLE TEMPORAL MIA"
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
        # BUILD
        # ----------------------------------------------------

        shadow_df = (
            build_controlled_dataset(
                dataset_name=
                    dataset_name,
                role="shadow",
                device=device,
            )
        )

        victim_df = (
            build_controlled_dataset(
                dataset_name=
                    dataset_name,
                role="victim",
                device=device,
            )
        )

        # ----------------------------------------------------
        # SAME FULL TEMPORAL WINDOW FOR BASELINE + T-MIA
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

        # Direct victim score diagnostic.
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

    # --------------------------------------------------------
    # FINAL TABLE
    # --------------------------------------------------------

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
        "tcomplex_same_triple_tmia_results.csv"
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
        "TCOMPLEX MIA / T-MIA RESULTS"
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
