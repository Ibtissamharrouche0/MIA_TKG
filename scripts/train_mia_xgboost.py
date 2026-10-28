from pathlib import Path

import pandas as pd

from xgboost import XGBClassifier

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
    "diff_prev_1",
    "diff_next_1",
    "abs_diff_prev_1",
    "abs_diff_next_1",
    "local_peak_1",

    "diff_prev_2",
    "diff_next_2",
    "abs_diff_prev_2",
    "abs_diff_next_2",
    "local_peak_2",

    "neighbor_asymmetry_1",
    "neighbor_asymmetry_2",
]

RAW_PLUS_DERIVED_FEATURES = (
    RAW_TEMPORAL_FEATURES + DERIVED_FEATURES
)


# ============================================================
# DERIVED FEATURES
# ============================================================

def add_derived_features(df):

    df = df.copy()

    s_m2 = df["score_t_minus_2"]
    s_m1 = df["score_t_minus_1"]
    s_0 = df["score_t"]
    s_p1 = df["score_t_plus_1"]
    s_p2 = df["score_t_plus_2"]

    df["diff_prev_1"] = s_0 - s_m1
    df["diff_next_1"] = s_0 - s_p1

    df["abs_diff_prev_1"] = (
        df["diff_prev_1"].abs()
    )

    df["abs_diff_next_1"] = (
        df["diff_next_1"].abs()
    )

    df["local_peak_1"] = (
        s_0 - ((s_m1 + s_p1) / 2.0)
    )

    df["diff_prev_2"] = s_0 - s_m2
    df["diff_next_2"] = s_0 - s_p2

    df["abs_diff_prev_2"] = (
        df["diff_prev_2"].abs()
    )

    df["abs_diff_next_2"] = (
        df["diff_next_2"].abs()
    )

    df["local_peak_2"] = (
        s_0
        -
        (
            s_m2
            + s_m1
            + s_p1
            + s_p2
        ) / 4.0
    )

    df["neighbor_asymmetry_1"] = (
        s_m1 - s_p1
    )

    df["neighbor_asymmetry_2"] = (
        s_m2 - s_p2
    )

    return df


# ============================================================
# CLEAN
# ============================================================

def clean_dataframe(df):

    # Same complete ±2 candidate set for all variants
    df = df.dropna(
        subset=RAW_TEMPORAL_FEATURES
    ).copy()

    df = add_derived_features(df)

    return df


# ============================================================
# XGBOOST ATTACK
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

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    # Train on shadow
    model.fit(
        X_shadow,
        y_shadow
    )

    # Evaluate on victim
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
    print("TEMPORAL MIA — XGBOOST")
    print("=" * 80)

    all_results = []

    for dataset in DATASETS:

        print("\n\n" + "=" * 80)
        print(dataset)
        print("=" * 80)

        attack_dir = ATTACK_ROOT / dataset

        shadow_path = (
            attack_dir
            / "shadow_attack_dataset.csv"
        )

        victim_path = (
            attack_dir
            / "victim_attack_dataset.csv"
        )

        # ====================================================
        # LOAD
        # ====================================================

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
        # CLEAN
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
            shadow_df,
            victim_df,
            BASELINE_FEATURES
        )

        # ====================================================
        # 2. RAW TEMPORAL
        # ====================================================

        raw_temporal = run_attack(
            shadow_df,
            victim_df,
            RAW_TEMPORAL_FEATURES
        )

        # ====================================================
        # 3. DERIVED ONLY
        # ====================================================

        derived_only = run_attack(
            shadow_df,
            victim_df,
            DERIVED_FEATURES
        )

        # ====================================================
        # 4. RAW + DERIVED
        # ====================================================

        raw_plus_derived = run_attack(
            shadow_df,
            victim_df,
            RAW_PLUS_DERIVED_FEATURES
        )

        # ====================================================
        # GAINS
        # ====================================================

        raw_gain = (
            raw_temporal["auc"]
            - baseline["auc"]
        )

        derived_gain = (
            derived_only["auc"]
            - baseline["auc"]
        )

        raw_plus_gain = (
            raw_plus_derived["auc"]
            - baseline["auc"]
        )

        raw_plus_vs_raw = (
            raw_plus_derived["auc"]
            - raw_temporal["auc"]
        )

        # ====================================================
        # PRINT
        # ====================================================

        def print_metrics(title, results):

            print(f"\n{title}")

            print(
                f"Accuracy : "
                f"{results['accuracy']:.6f}"
            )

            print(
                f"Precision: "
                f"{results['precision']:.6f}"
            )

            print(
                f"Recall   : "
                f"{results['recall']:.6f}"
            )

            print(
                f"F1       : "
                f"{results['f1']:.6f}"
            )

            print(
                f"ROC-AUC  : "
                f"{results['auc']:.6f}"
            )

        print_metrics(
            "XGBoost — Baseline",
            baseline
        )

        print_metrics(
            "XGBoost — Raw Temporal",
            raw_temporal
        )

        print_metrics(
            "XGBoost — Derived Only",
            derived_only
        )

        print_metrics(
            "XGBoost — Raw + Derived",
            raw_plus_derived
        )

        print("\nAUC gains")

        print(
            "Raw Temporal - Baseline : "
            f"{raw_gain:+.6f}"
        )

        print(
            "Derived Only - Baseline : "
            f"{derived_gain:+.6f}"
        )

        print(
            "Raw+Derived - Baseline  : "
            f"{raw_plus_gain:+.6f}"
        )

        print(
            "Raw+Derived - Raw       : "
            f"{raw_plus_vs_raw:+.6f}"
        )

        # ====================================================
        # STORE
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

                "raw_gain_vs_baseline":
                    raw_gain,

                "derived_gain_vs_baseline":
                    derived_gain,

                "raw_plus_derived_gain_vs_baseline":
                    raw_plus_gain,

                "raw_plus_derived_gain_vs_raw":
                    raw_plus_vs_raw,
            }
        )

    # ========================================================
    # SAVE
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
        / "mia_xgboost_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n\n" + "=" * 80)
    print("SUMMARY — XGBOOST")
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