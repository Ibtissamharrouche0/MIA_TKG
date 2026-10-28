from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# CONFIG
# ============================================================

ATTACK_ROOT = Path("/home/harrouche/Paper2/attack_data")
RESULTS_ROOT = Path("/home/harrouche/Paper2/results")

DATASETS = [
    "ICEWS05-15",
    "ICEWS14",
    "YAGO11k",
    "WIKIDATA12k",
]

SEEDS = [
    42,
    123,
    2025,
    7,
    99,
]

TEMPORAL_FEATURES = [
    "score_t_minus_2",
    "score_t_minus_1",
    "score_t",
    "score_t_plus_1",
    "score_t_plus_2",
]

BASELINE_FEATURES = [
    "score_t"
]


# ============================================================
# BALANCE DATASET USING A GIVEN SEED
# ============================================================

def balance_dataframe(df, seed):

    members = df[
        df["label"] == 1
    ]

    nonmembers = df[
        df["label"] == 0
    ]

    n = min(
        len(members),
        len(nonmembers)
    )

    members = members.sample(
        n=n,
        random_state=seed
    )

    nonmembers = nonmembers.sample(
        n=n,
        random_state=seed
    )

    balanced = pd.concat(
        [
            members,
            nonmembers
        ],
        ignore_index=True
    )

    balanced = balanced.sample(
        frac=1,
        random_state=seed
    ).reset_index(
        drop=True
    )

    return balanced


# ============================================================
# TRAIN + EVALUATE ATTACK
# ============================================================

def evaluate_attack(
    shadow_df,
    victim_df,
    features,
    seed
):

    X_shadow = shadow_df[
        features
    ].values

    y_shadow = shadow_df[
        "label"
    ].values

    X_victim = victim_df[
        features
    ].values

    y_victim = victim_df[
        "label"
    ].values

    # --------------------------------------------------------
    # Attack model
    # --------------------------------------------------------

    attack_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Train on SHADOW
    # --------------------------------------------------------

    attack_model.fit(
        X_shadow,
        y_shadow
    )

    # --------------------------------------------------------
    # Evaluate on VICTIM
    # --------------------------------------------------------

    y_pred = attack_model.predict(
        X_victim
    )

    y_prob = attack_model.predict_proba(
        X_victim
    )[:, 1]

    return {
        "accuracy": accuracy_score(
            y_victim,
            y_pred
        ),

        "precision": precision_score(
            y_victim,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_victim,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_victim,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_victim,
            y_prob
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    print("=" * 80)
    print("MULTI-SEED MEMBERSHIP INFERENCE ATTACK")
    print("=" * 80)

    # ========================================================
    # DATASETS
    # ========================================================

    for dataset in DATASETS:

        print("\n" + "=" * 80)
        print(f"DATASET: {dataset}")
        print("=" * 80)

        dataset_dir = (
            ATTACK_ROOT
            /
            dataset
        )

        shadow_path = (
            dataset_dir
            /
            "shadow_attack_dataset.csv"
        )

        victim_path = (
            dataset_dir
            /
            "victim_attack_dataset.csv"
        )

        # ----------------------------------------------------
        # Load attack datasets
        # ----------------------------------------------------

        shadow_df = pd.read_csv(
            shadow_path
        )

        victim_df = pd.read_csv(
            victim_path
        )

        print(
            f"Original shadow samples : "
            f"{len(shadow_df)}"
        )

        print(
            f"Original victim samples : "
            f"{len(victim_df)}"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Use same complete temporal window for baseline
        # and Temporal-MIA.
        # ----------------------------------------------------

        shadow_df = shadow_df.dropna(
            subset=TEMPORAL_FEATURES
        ).reset_index(
            drop=True
        )

        victim_df = victim_df.dropna(
            subset=TEMPORAL_FEATURES
        ).reset_index(
            drop=True
        )

        print(
            f"Clean shadow samples    : "
            f"{len(shadow_df)}"
        )

        print(
            f"Clean victim samples    : "
            f"{len(victim_df)}"
        )

        # ====================================================
        # MULTIPLE SEEDS
        # ====================================================

        for seed in SEEDS:

            print(
                f"\n--- Seed {seed} ---"
            )

            # ------------------------------------------------
            # Rebalance / resample
            # ------------------------------------------------

            shadow_seed = balance_dataframe(
                shadow_df,
                seed
            )

            victim_seed = balance_dataframe(
                victim_df,
                seed
            )

            # =================================================
            # BASELINE
            # =================================================

            baseline = evaluate_attack(
                shadow_df=shadow_seed,
                victim_df=victim_seed,
                features=BASELINE_FEATURES,
                seed=seed
            )

            # =================================================
            # TEMPORAL
            # =================================================

            temporal = evaluate_attack(
                shadow_df=shadow_seed,
                victim_df=victim_seed,
                features=TEMPORAL_FEATURES,
                seed=seed
            )

            auc_gain = (
                temporal["roc_auc"]
                -
                baseline["roc_auc"]
            )

            print(
                f"Baseline AUC : "
                f"{baseline['roc_auc']:.6f}"
            )

            print(
                f"Temporal AUC : "
                f"{temporal['roc_auc']:.6f}"
            )

            print(
                f"AUC gain     : "
                f"{auc_gain:+.6f}"
            )

            # ------------------------------------------------
            # Save baseline
            # ------------------------------------------------

            all_results.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "attack": "Baseline-MIA",
                    **baseline
                }
            )

            # ------------------------------------------------
            # Save temporal
            # ------------------------------------------------

            all_results.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "attack": "Temporal-MIA",
                    **temporal
                }
            )

    # ========================================================
    # RAW RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    raw_path = (
        RESULTS_ROOT
        /
        "mia_multi_seed_raw.csv"
    )

    results_df.to_csv(
        raw_path,
        index=False
    )

    # ========================================================
    # MEAN +/- STD
    # ========================================================

    summary = (
        results_df
        .groupby(
            [
                "dataset",
                "attack"
            ]
        )
        .agg(
            {
                "accuracy": [
                    "mean",
                    "std"
                ],
                "precision": [
                    "mean",
                    "std"
                ],
                "recall": [
                    "mean",
                    "std"
                ],
                "f1": [
                    "mean",
                    "std"
                ],
                "roc_auc": [
                    "mean",
                    "std"
                ]
            }
        )
    )

    # Flatten column names
    summary.columns = [
        "_".join(col)
        for col in summary.columns
    ]

    summary = summary.reset_index()

    summary_path = (
        RESULTS_ROOT
        /
        "mia_multi_seed_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # AUC GAIN PER SEED
    # ========================================================

    baseline_auc = (
        results_df[
            results_df["attack"]
            ==
            "Baseline-MIA"
        ][
            [
                "dataset",
                "seed",
                "roc_auc"
            ]
        ]
        .rename(
            columns={
                "roc_auc":
                    "baseline_auc"
            }
        )
    )

    temporal_auc = (
        results_df[
            results_df["attack"]
            ==
            "Temporal-MIA"
        ][
            [
                "dataset",
                "seed",
                "roc_auc"
            ]
        ]
        .rename(
            columns={
                "roc_auc":
                    "temporal_auc"
            }
        )
    )

    gains = baseline_auc.merge(
        temporal_auc,
        on=[
            "dataset",
            "seed"
        ]
    )

    gains["auc_gain"] = (
        gains["temporal_auc"]
        -
        gains["baseline_auc"]
    )

    gain_summary = (
        gains
        .groupby(
            "dataset"
        )["auc_gain"]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max"
            ]
        )
        .reset_index()
    )

    gain_path = (
        RESULTS_ROOT
        /
        "mia_multi_seed_auc_gain.csv"
    )

    gain_summary.to_csv(
        gain_path,
        index=False
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n" + "=" * 80)
    print("FINAL MULTI-SEED SUMMARY")
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("TEMPORAL AUC GAIN SUMMARY")
    print("=" * 80)

    print(
        gain_summary.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(raw_path)
    print(summary_path)
    print(gain_path)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()