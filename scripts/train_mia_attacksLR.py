from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

ATTACK_ROOT = Path(
    "/home/harrouche/Paper2/attack_data"
)

RESULT_ROOT = Path(
    "/home/harrouche/Paper2/results"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


DATASETS = [
    "ICEWS05-15",
    "ICEWS14",
    "YAGO11k",
    "WIKIDATA12k",
]


BASELINE_FEATURES = [
    "score_t"
]


TEMPORAL_FEATURES = [
    "score_t_minus_2",
    "score_t_minus_1",
    "score_t",
    "score_t_plus_1",
    "score_t_plus_2",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_attack_data(dataset_name):

    dataset_dir = (
        ATTACK_ROOT
        /
        dataset_name
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

    shadow_df = pd.read_csv(
        shadow_path
    )

    victim_df = pd.read_csv(
        victim_path
    )

    return (
        shadow_df,
        victim_df
    )


# ============================================================
# REMOVE TEMPORAL NaNs
# ============================================================

def clean_temporal_data(df):

    return (
        df
        .dropna(
            subset=TEMPORAL_FEATURES
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# TRAIN + EVALUATE ONE ATTACK
# ============================================================

def run_attack(
    shadow_df,
    victim_df,
    features,
    attack_name
):

    # --------------------------------------------------------
    # Training data = shadow
    # --------------------------------------------------------

    X_train = shadow_df[
        features
    ].values

    y_train = shadow_df[
        "label"
    ].values

    # --------------------------------------------------------
    # Evaluation data = victim
    # --------------------------------------------------------

    X_test = victim_df[
        features
    ].values

    y_test = victim_df[
        "label"
    ].values

    # --------------------------------------------------------
    # Attack classifier
    # --------------------------------------------------------

    classifier = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Train on SHADOW
    # --------------------------------------------------------

    classifier.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predict on VICTIM
    # --------------------------------------------------------

    y_pred = classifier.predict(
        X_test
    )

    y_prob = classifier.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {

        "attack":
            attack_name,

        "accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_test,
                y_prob
            ),

        "shadow_samples":
            len(shadow_df),

        "victim_samples":
            len(victim_df),
    }

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 80
    )

    print(
        "MEMBERSHIP INFERENCE ATTACK EVALUATION"
    )

    print(
        "=" * 80
    )

    all_results = []

    # ========================================================
    # DATASET LOOP
    # ========================================================

    for dataset_name in DATASETS:

        print(
            "\n"
            +
            "=" * 80
        )

        print(
            f"DATASET: {dataset_name}"
        )

        print(
            "=" * 80
        )

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        (
            shadow_df,
            victim_df
        ) = load_attack_data(
            dataset_name
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
        # use same cleaned examples for BOTH attacks
        # ----------------------------------------------------

        shadow_clean = (
            clean_temporal_data(
                shadow_df
            )
        )

        victim_clean = (
            clean_temporal_data(
                victim_df
            )
        )

        print(
            f"Clean shadow samples    : "
            f"{len(shadow_clean)}"
        )

        print(
            f"Clean victim samples    : "
            f"{len(victim_clean)}"
        )

        # ====================================================
        # BASELINE MIA
        # ====================================================

        baseline_results = run_attack(

            shadow_df=shadow_clean,

            victim_df=victim_clean,

            features=BASELINE_FEATURES,

            attack_name="Baseline-MIA"
        )

        baseline_results[
            "dataset"
        ] = dataset_name

        # ====================================================
        # TEMPORAL MIA
        # ====================================================

        temporal_results = run_attack(

            shadow_df=shadow_clean,

            victim_df=victim_clean,

            features=TEMPORAL_FEATURES,

            attack_name="Temporal-MIA"
        )

        temporal_results[
            "dataset"
        ] = dataset_name

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_results.append(
            baseline_results
        )

        all_results.append(
            temporal_results
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            "\nBaseline MIA"
        )

        print(
            f"Accuracy : "
            f"{baseline_results['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{baseline_results['precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{baseline_results['recall']:.4f}"
        )

        print(
            f"F1       : "
            f"{baseline_results['f1']:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{baseline_results['roc_auc']:.4f}"
        )

        print(
            "\nTemporal MIA"
        )

        print(
            f"Accuracy : "
            f"{temporal_results['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{temporal_results['precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{temporal_results['recall']:.4f}"
        )

        print(
            f"F1       : "
            f"{temporal_results['f1']:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{temporal_results['roc_auc']:.4f}"
        )

        # ----------------------------------------------------
        # Improvement
        # ----------------------------------------------------

        auc_gain = (
            temporal_results[
                "roc_auc"
            ]
            -
            baseline_results[
                "roc_auc"
            ]
        )

        print(
            "\nTemporal AUC gain:"
        )

        print(
            f"{auc_gain:+.4f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df[
        [
            "dataset",
            "attack",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "shadow_samples",
            "victim_samples"
        ]
    ]

    output_path = (
        RESULT_ROOT
        /
        "mia_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 80
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