from pathlib import Path

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


# ============================================================
# TEMPORAL WINDOWS
# ============================================================

WINDOWS = {

    0: [
        "score_t"
    ],

    1: [
        "score_t_minus_1",
        "score_t",
        "score_t_plus_1"
    ],

    2: [
        "score_t_minus_2",
        "score_t_minus_1",
        "score_t",
        "score_t_plus_1",
        "score_t_plus_2"
    ]
}


ALL_TEMPORAL_FEATURES = [
    "score_t_minus_2",
    "score_t_minus_1",
    "score_t",
    "score_t_plus_1",
    "score_t_plus_2"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data(dataset_name):

    dataset_dir = (
        ATTACK_ROOT
        /
        dataset_name
    )

    shadow_df = pd.read_csv(
        dataset_dir
        /
        "shadow_attack_dataset.csv"
    )

    victim_df = pd.read_csv(
        dataset_dir
        /
        "victim_attack_dataset.csv"
    )

    return shadow_df, victim_df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_for_window(
    df,
    features
):

    return (
        df
        .dropna(
            subset=features
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# RUN ONE ATTACK
# ============================================================

def run_attack(
    shadow_df,
    victim_df,
    features
):

    X_train = shadow_df[
        features
    ].values

    y_train = shadow_df[
        "label"
    ].values

    X_test = victim_df[
        features
    ].values

    y_test = victim_df[
        "label"
    ].values

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

    classifier.fit(
        X_train,
        y_train
    )

    y_pred = classifier.predict(
        X_test
    )

    y_prob = classifier.predict_proba(
        X_test
    )[:, 1]

    return {

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
            len(victim_df)
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 85
    )

    print(
        "TEMPORAL WINDOW ABLATION"
    )

    print(
        "=" * 85
    )

    all_results = []

    # ========================================================
    # DATASET LOOP
    # ========================================================

    for dataset_name in DATASETS:

        print(
            "\n"
            +
            "=" * 85
        )

        print(
            f"DATASET: {dataset_name}"
        )

        print(
            "=" * 85
        )

        shadow_df, victim_df = load_data(
            dataset_name
        )

        # ====================================================
        # IMPORTANT:
        # Fair comparison between Δ=0,1,2
        #
        # We first remove rows that contain ANY NaN in the
        # complete Δ=2 window.
        #
        # Thus all windows use EXACTLY the same candidates.
        # ====================================================

        shadow_common = (
            shadow_df
            .dropna(
                subset=ALL_TEMPORAL_FEATURES
            )
            .reset_index(
                drop=True
            )
        )

        victim_common = (
            victim_df
            .dropna(
                subset=ALL_TEMPORAL_FEATURES
            )
            .reset_index(
                drop=True
            )
        )

        print(
            f"Common shadow samples : "
            f"{len(shadow_common)}"
        )

        print(
            f"Common victim samples : "
            f"{len(victim_common)}"
        )

        # ====================================================
        # WINDOW LOOP
        # ====================================================

        for delta, features in WINDOWS.items():

            results = run_attack(

                shadow_df=
                    shadow_common,

                victim_df=
                    victim_common,

                features=
                    features
            )

            result_row = {

                "dataset":
                    dataset_name,

                "delta":
                    delta,

                "num_features":
                    len(features),

                **results
            }

            all_results.append(
                result_row
            )

            print()

            print(
                f"Delta = {delta}"
            )

            print(
                f"Features : {features}"
            )

            print(
                f"Accuracy : "
                f"{results['accuracy']:.4f}"
            )

            print(
                f"F1       : "
                f"{results['f1']:.4f}"
            )

            print(
                f"ROC-AUC  : "
                f"{results['roc_auc']:.4f}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    output_path = (
        RESULT_ROOT
        /
        "temporal_window_ablation.csv"
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
        "=" * 85
    )

    print(
        "FINAL TEMPORAL WINDOW ABLATION"
    )

    print(
        "=" * 85
    )

    display_cols = [
        "dataset",
        "delta",
        "num_features",
        "accuracy",
        "f1",
        "roc_auc"
    ]

    print(
        results_df[
            display_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )