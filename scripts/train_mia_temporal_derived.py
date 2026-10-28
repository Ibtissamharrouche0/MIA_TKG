from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

ATTACK_ROOT = Path("/home/harrouche/Paper2/attack_data")
RESULTS_ROOT = Path("/home/harrouche/Paper2/results")

DATASETS = [
    "ICEWS05-15",
    "ICEWS14",
    "YAGO11k",
    "WIKIDATA12k",
]


# ============================================================
# FEATURES
# ============================================================

BASELINE_FEATURES = [
    "score_t",
]


RAW_TEMPORAL_FEATURES = [
    "score_t_minus_2",
    "score_t_minus_1",
    "score_t",
    "score_t_plus_1",
    "score_t_plus_2",
]


DERIVED_FEATURES = [
    # Immediate differences
    "diff_prev_1",
    "diff_next_1",

    # Absolute immediate differences
    "abs_diff_prev_1",
    "abs_diff_next_1",

    # Local peak around t
    "local_peak_1",

    # Wider differences
    "diff_prev_2",
    "diff_next_2",

    # Absolute wider differences
    "abs_diff_prev_2",
    "abs_diff_next_2",

    # Wider local peak
    "local_peak_2",

    # Past/future asymmetry
    "neighbor_asymmetry_1",
    "neighbor_asymmetry_2",
]


RAW_PLUS_DERIVED_FEATURES = (
    RAW_TEMPORAL_FEATURES
    +
    DERIVED_FEATURES
)


# ============================================================
# ADD DERIVED TEMPORAL FEATURES
# ============================================================

def add_derived_features(df):

    df = df.copy()

    s_m2 = df["score_t_minus_2"]
    s_m1 = df["score_t_minus_1"]
    s_0 = df["score_t"]
    s_p1 = df["score_t_plus_1"]
    s_p2 = df["score_t_plus_2"]

    # --------------------------------------------------------
    # Immediate differences
    #
    # s(t) - s(t-1)
    # s(t) - s(t+1)
    # --------------------------------------------------------

    df["diff_prev_1"] = (
        s_0 - s_m1
    )

    df["diff_next_1"] = (
        s_0 - s_p1
    )

    # --------------------------------------------------------
    # Absolute immediate differences
    # --------------------------------------------------------

    df["abs_diff_prev_1"] = (
        df["diff_prev_1"].abs()
    )

    df["abs_diff_next_1"] = (
        df["diff_next_1"].abs()
    )

    # --------------------------------------------------------
    # Local peak ±1
    #
    # s(t) - mean(s(t-1), s(t+1))
    # --------------------------------------------------------

    df["local_peak_1"] = (
        s_0
        -
        (
            s_m1
            +
            s_p1
        ) / 2.0
    )

    # --------------------------------------------------------
    # Wider differences
    #
    # s(t) - s(t-2)
    # s(t) - s(t+2)
    # --------------------------------------------------------

    df["diff_prev_2"] = (
        s_0 - s_m2
    )

    df["diff_next_2"] = (
        s_0 - s_p2
    )

    # --------------------------------------------------------
    # Absolute wider differences
    # --------------------------------------------------------

    df["abs_diff_prev_2"] = (
        df["diff_prev_2"].abs()
    )

    df["abs_diff_next_2"] = (
        df["diff_next_2"].abs()
    )

    # --------------------------------------------------------
    # Wider local peak ±2
    #
    # s(t) - mean(
    #     s(t-2),
    #     s(t-1),
    #     s(t+1),
    #     s(t+2)
    # )
    # --------------------------------------------------------

    df["local_peak_2"] = (
        s_0
        -
        (
            s_m2
            +
            s_m1
            +
            s_p1
            +
            s_p2
        ) / 4.0
    )

    # --------------------------------------------------------
    # Past/future asymmetry
    # --------------------------------------------------------

    df["neighbor_asymmetry_1"] = (
        s_m1 - s_p1
    )

    df["neighbor_asymmetry_2"] = (
        s_m2 - s_p2
    )

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_dataframe(df):

    # IMPORTANT:
    # Same candidate rows for baseline, raw temporal,
    # derived and raw+derived.
    #
    # We remove every row that does not have the complete
    # temporal window ±2.

    df = df.dropna(
        subset=RAW_TEMPORAL_FEATURES
    ).copy()

    df = add_derived_features(
        df
    )

    return df


# ============================================================
# TRAIN + EVALUATE ATTACK
# ============================================================

def run_attack(
    shadow_df,
    victim_df,
    features
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
    # Logistic Regression attack classifier
    # --------------------------------------------------------

    model = Pipeline(
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
    # Train on SHADOW attack data
    # --------------------------------------------------------

    model.fit(
        X_shadow,
        y_shadow
    )

    # --------------------------------------------------------
    # Evaluate on VICTIM attack data
    # --------------------------------------------------------

    y_pred = model.predict(
        X_victim
    )

    y_prob = model.predict_proba(
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

        "auc": roc_auc_score(
            y_victim,
            y_prob
        )
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("TEMPORAL MIA — DERIVED TEMPORAL FEATURES")
    print("=" * 80)

    # Store all results here
    all_results = []

    for dataset in DATASETS:

        print("\n\n" + "=" * 80)
        print(dataset)
        print("=" * 80)

        attack_dir = (
            ATTACK_ROOT
            /
            dataset
        )

        # ====================================================
        # LOAD CURRENT RANDOM-SAMPLING ATTACK DATASETS
        # ====================================================

        shadow_path = (
            attack_dir
            /
            "shadow_attack_dataset.csv"
        )

        victim_path = (
            attack_dir
            /
            "victim_attack_dataset.csv"
        )

        shadow_df = pd.read_csv(
            shadow_path
        )

        victim_df = pd.read_csv(
            victim_path
        )

        original_shadow_samples = len(
            shadow_df
        )

        original_victim_samples = len(
            victim_df
        )

        print(
            f"Original shadow samples : "
            f"{original_shadow_samples}"
        )

        print(
            f"Original victim samples : "
            f"{original_victim_samples}"
        )

        # ====================================================
        # CLEAN + DERIVED FEATURES
        # ====================================================

        shadow_df = clean_dataframe(
            shadow_df
        )

        victim_df = clean_dataframe(
            victim_df
        )

        clean_shadow_samples = len(
            shadow_df
        )

        clean_victim_samples = len(
            victim_df
        )

        print(
            f"Clean shadow samples    : "
            f"{clean_shadow_samples}"
        )

        print(
            f"Clean victim samples    : "
            f"{clean_victim_samples}"
        )

        # ====================================================
        # 1. BASELINE
        # ====================================================

        baseline = run_attack(
            shadow_df=shadow_df,
            victim_df=victim_df,
            features=BASELINE_FEATURES
        )

        # ====================================================
        # 2. RAW TEMPORAL
        # ====================================================

        raw_temporal = run_attack(
            shadow_df=shadow_df,
            victim_df=victim_df,
            features=RAW_TEMPORAL_FEATURES
        )

        # ====================================================
        # 3. DERIVED FEATURES ONLY
        # ====================================================

        derived_only = run_attack(
            shadow_df=shadow_df,
            victim_df=victim_df,
            features=DERIVED_FEATURES
        )

        # ====================================================
        # 4. RAW + DERIVED
        # ====================================================

        raw_plus_derived = run_attack(
            shadow_df=shadow_df,
            victim_df=victim_df,
            features=RAW_PLUS_DERIVED_FEATURES
        )

        # ====================================================
        # GAINS
        # ====================================================

        raw_gain_vs_baseline = (
            raw_temporal["auc"]
            -
            baseline["auc"]
        )

        derived_gain_vs_baseline = (
            derived_only["auc"]
            -
            baseline["auc"]
        )

        raw_plus_derived_gain_vs_baseline = (
            raw_plus_derived["auc"]
            -
            baseline["auc"]
        )

        raw_plus_derived_gain_vs_raw = (
            raw_plus_derived["auc"]
            -
            raw_temporal["auc"]
        )

        # ====================================================
        # PRINT
        # ====================================================

        print("\nBaseline MIA")

        print(
            f"Accuracy : "
            f"{baseline['accuracy']:.6f}"
        )

        print(
            f"Precision: "
            f"{baseline['precision']:.6f}"
        )

        print(
            f"Recall   : "
            f"{baseline['recall']:.6f}"
        )

        print(
            f"F1       : "
            f"{baseline['f1']:.6f}"
        )

        print(
            f"ROC-AUC  : "
            f"{baseline['auc']:.6f}"
        )

        # ----------------------------------------------------

        print("\nRaw Temporal MIA")

        print(
            f"Accuracy : "
            f"{raw_temporal['accuracy']:.6f}"
        )

        print(
            f"Precision: "
            f"{raw_temporal['precision']:.6f}"
        )

        print(
            f"Recall   : "
            f"{raw_temporal['recall']:.6f}"
        )

        print(
            f"F1       : "
            f"{raw_temporal['f1']:.6f}"
        )

        print(
            f"ROC-AUC  : "
            f"{raw_temporal['auc']:.6f}"
        )

        # ----------------------------------------------------

        print("\nDerived Features Only")

        print(
            f"Accuracy : "
            f"{derived_only['accuracy']:.6f}"
        )

        print(
            f"Precision: "
            f"{derived_only['precision']:.6f}"
        )

        print(
            f"Recall   : "
            f"{derived_only['recall']:.6f}"
        )

        print(
            f"F1       : "
            f"{derived_only['f1']:.6f}"
        )

        print(
            f"ROC-AUC  : "
            f"{derived_only['auc']:.6f}"
        )

        # ----------------------------------------------------

        print("\nRaw + Derived Features")

        print(
            f"Accuracy : "
            f"{raw_plus_derived['accuracy']:.6f}"
        )

        print(
            f"Precision: "
            f"{raw_plus_derived['precision']:.6f}"
        )

        print(
            f"Recall   : "
            f"{raw_plus_derived['recall']:.6f}"
        )

        print(
            f"F1       : "
            f"{raw_plus_derived['f1']:.6f}"
        )

        print(
            f"ROC-AUC  : "
            f"{raw_plus_derived['auc']:.6f}"
        )

        # ----------------------------------------------------

        print("\nAUC gains")

        print(
            "Raw Temporal - Baseline : "
            f"{raw_gain_vs_baseline:+.6f}"
        )

        print(
            "Derived Only - Baseline : "
            f"{derived_gain_vs_baseline:+.6f}"
        )

        print(
            "Raw+Derived - Baseline  : "
            f"{raw_plus_derived_gain_vs_baseline:+.6f}"
        )

        print(
            "Raw+Derived - Raw       : "
            f"{raw_plus_derived_gain_vs_raw:+.6f}"
        )

        # ====================================================
        # STORE RESULTS
        # ====================================================

        all_results.append(
            {
                "dataset": dataset,

                "original_shadow_samples":
                    original_shadow_samples,

                "original_victim_samples":
                    original_victim_samples,

                "clean_shadow_samples":
                    clean_shadow_samples,

                "clean_victim_samples":
                    clean_victim_samples,

                # Baseline
                "baseline_accuracy":
                    baseline["accuracy"],

                "baseline_precision":
                    baseline["precision"],

                "baseline_recall":
                    baseline["recall"],

                "baseline_f1":
                    baseline["f1"],

                "baseline_auc":
                    baseline["auc"],

                # Raw temporal
                "raw_temporal_accuracy":
                    raw_temporal["accuracy"],

                "raw_temporal_precision":
                    raw_temporal["precision"],

                "raw_temporal_recall":
                    raw_temporal["recall"],

                "raw_temporal_f1":
                    raw_temporal["f1"],

                "raw_temporal_auc":
                    raw_temporal["auc"],

                # Derived only
                "derived_only_accuracy":
                    derived_only["accuracy"],

                "derived_only_precision":
                    derived_only["precision"],

                "derived_only_recall":
                    derived_only["recall"],

                "derived_only_f1":
                    derived_only["f1"],

                "derived_only_auc":
                    derived_only["auc"],

                # Raw + derived
                "raw_plus_derived_accuracy":
                    raw_plus_derived["accuracy"],

                "raw_plus_derived_precision":
                    raw_plus_derived["precision"],

                "raw_plus_derived_recall":
                    raw_plus_derived["recall"],

                "raw_plus_derived_f1":
                    raw_plus_derived["f1"],

                "raw_plus_derived_auc":
                    raw_plus_derived["auc"],

                # Gains
                "raw_gain_vs_baseline":
                    raw_gain_vs_baseline,

                "derived_gain_vs_baseline":
                    derived_gain_vs_baseline,

                "raw_plus_derived_gain_vs_baseline":
                    raw_plus_derived_gain_vs_baseline,

                "raw_plus_derived_gain_vs_raw":
                    raw_plus_derived_gain_vs_raw,
            }
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df = pd.DataFrame(
        all_results
    )

    output_path = (
        RESULTS_ROOT
        /
        "mia_temporal_derived_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary_columns = [
        "dataset",
        "baseline_auc",
        "raw_temporal_auc",
        "derived_only_auc",
        "raw_plus_derived_auc",
        "raw_plus_derived_gain_vs_raw",
    ]

    print(
        results_df[
            summary_columns
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)

    print(
        f"Results saved to:\n"
        f"{output_path}"
    )

    print("=" * 80)
    print("FINISHED")
    print("=" * 80)