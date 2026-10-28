from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
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

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 512
MAX_EPOCHS = 200
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

VALIDATION_SIZE = 0.20


# ============================================================
# FEATURES
# ============================================================

SCORE_FEATURES = [
    "score_t",
]

TEMPORAL_FEATURES = [
    "score_t_minus_2",
    "score_t_minus_1",
    "score_t",
    "score_t_plus_1",
    "score_t_plus_2",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MLP
# ============================================================

class AttackMLP(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.network(x).squeeze(1)


# ============================================================
# DATASET
# ============================================================

class NumpyDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim
):

    model = AttackMLP(
        input_dim=input_dim
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    train_dataset = NumpyDataset(
        X_train,
        y_train
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
        device=DEVICE
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32,
        device=DEVICE
    )

    best_auc = -np.inf
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        total_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                *
                len(y_batch)
            )

        train_loss = (
            total_loss
            /
            len(train_dataset)
        )

        # ----------------------------------------------------
        # SHADOW VALIDATION
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_val_tensor
            )

            val_probs = torch.sigmoid(
                val_logits
            ).cpu().numpy()

        val_auc = roc_auc_score(
            y_val,
            val_probs
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if val_auc > best_auc + 1e-5:

            best_auc = val_auc

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            patience_counter = 0

        else:

            patience_counter += 1

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Train loss={train_loss:.6f} | "
                f"Shadow val AUC={val_auc:.6f}"
            )

        if patience_counter >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch}"
            )

            break

    model.load_state_dict(
        best_state
    )

    print(
        f"Best shadow validation AUC: "
        f"{best_auc:.6f} "
        f"(epoch {best_epoch})"
    )

    return model, best_auc, best_epoch


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y
):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        logits = model(
            X_tensor
        )

        probs = torch.sigmoid(
            logits
        ).cpu().numpy()

    preds = (
        probs >= 0.5
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y,
            preds
        ),

        "precision": precision_score(
            y,
            preds,
            zero_division=0
        ),

        "recall": recall_score(
            y,
            preds,
            zero_division=0
        ),

        "f1": f1_score(
            y,
            preds,
            zero_division=0
        ),

        "auc": roc_auc_score(
            y,
            probs
        )
    }


# ============================================================
# RUN ONE FEATURE SET
# ============================================================

def run_experiment(
    shadow_df,
    victim_df,
    features,
    experiment_name
):

    print(
        "\n" + "-" * 70
    )

    print(
        experiment_name
    )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # SHADOW DATA
    # --------------------------------------------------------

    X_shadow = shadow_df[
        features
    ].values.astype(
        np.float32
    )

    y_shadow = shadow_df[
        "label"
    ].values.astype(
        np.int64
    )

    # --------------------------------------------------------
    # VICTIM DATA
    # --------------------------------------------------------

    X_victim = victim_df[
        features
    ].values.astype(
        np.float32
    )

    y_victim = victim_df[
        "label"
    ].values.astype(
        np.int64
    )

    # ========================================================
    # SPLIT SHADOW INTO TRAIN / VALIDATION
    # ========================================================

    (
        X_train,
        X_val,
        y_train,
        y_val
    ) = train_test_split(
        X_shadow,
        y_shadow,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=y_shadow
    )

    # ========================================================
    # STANDARDIZATION
    # IMPORTANT:
    # FIT ONLY ON SHADOW TRAIN
    # ========================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_victim_scaled = scaler.transform(
        X_victim
    )

    # ========================================================
    # TRAIN
    # ========================================================

    model, val_auc, best_epoch = train_mlp(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=len(features)
    )

    # ========================================================
    # VICTIM EVALUATION
    # ========================================================

    result = evaluate_model(
        model,
        X_victim_scaled,
        y_victim
    )

    result[
        "shadow_val_auc"
    ] = val_auc

    result[
        "best_epoch"
    ] = best_epoch

    return result


# ============================================================
# PRINT RESULTS
# ============================================================

def print_metrics(
    title,
    result
):

    print(
        f"\n{title}"
    )

    print(
        f"Accuracy       : "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"Precision      : "
        f"{result['precision']:.6f}"
    )

    print(
        f"Recall         : "
        f"{result['recall']:.6f}"
    )

    print(
        f"F1             : "
        f"{result['f1']:.6f}"
    )

    print(
        f"Victim ROC-AUC : "
        f"{result['auc']:.6f}"
    )

    print(
        f"Shadow Val AUC : "
        f"{result['shadow_val_auc']:.6f}"
    )

    print(
        f"Best Epoch     : "
        f"{result['best_epoch']}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    set_seed(
        SEED
    )

    print(
        "=" * 80
    )

    print(
        "MLP MIA — SCORE-ONLY vs RAW TEMPORAL"
    )

    print(
        "=" * 80
    )

    print(
        f"Device: {DEVICE}"
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    # ========================================================
    # LOOP DATASETS
    # ========================================================

    for dataset in DATASETS:

        print(
            "\n\n" + "=" * 80
        )

        print(
            dataset
        )

        print(
            "=" * 80
        )

        attack_dir = (
            ATTACK_ROOT
            / dataset
        )

        shadow_path = (
            attack_dir
            / "shadow_attack_dataset.csv"
        )

        victim_path = (
            attack_dir
            / "victim_attack_dataset.csv"
        )

        # ----------------------------------------------------
        # LOAD
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

        # ====================================================
        # SAME CLEAN CANDIDATES FOR BOTH EXPERIMENTS
        #
        # Drop rows missing any temporal neighbor.
        #
        # This keeps Score-only and Temporal evaluation fair.
        # ====================================================

        shadow_df = shadow_df.dropna(
            subset=TEMPORAL_FEATURES
        ).copy()

        victim_df = victim_df.dropna(
            subset=TEMPORAL_FEATURES
        ).copy()

        print(
            f"Clean shadow samples    : "
            f"{len(shadow_df)}"
        )

        print(
            f"Clean victim samples    : "
            f"{len(victim_df)}"
        )

        # ====================================================
        # 1. SCORE ONLY
        # ====================================================

        set_seed(
            SEED
        )

        score_result = run_experiment(
            shadow_df=shadow_df,
            victim_df=victim_df,
            features=SCORE_FEATURES,
            experiment_name=(
                "1. Score-only MLP"
            )
        )

        # ====================================================
        # 2. TEMPORAL
        # ====================================================

        set_seed(
            SEED
        )

        temporal_result = run_experiment(
            shadow_df=shadow_df,
            victim_df=victim_df,
            features=TEMPORAL_FEATURES,
            experiment_name=(
                "2. Raw Temporal MLP"
            )
        )

        # ====================================================
        # PRINT
        # ====================================================

        print_metrics(
            "1. Score-only MLP",
            score_result
        )

        print_metrics(
            "2. Raw Temporal MLP",
            temporal_result
        )

        gain = (
            temporal_result["auc"]
            -
            score_result["auc"]
        )

        print(
            "\nAUC gain"
        )

        print(
            f"Temporal - Score : "
            f"{gain:+.6f}"
        )

        # ====================================================
        # SAVE
        # ====================================================

        all_results.append(
            {
                "dataset": dataset,

                "shadow_samples":
                    len(shadow_df),

                "victim_samples":
                    len(victim_df),

                "score_accuracy":
                    score_result["accuracy"],

                "score_precision":
                    score_result["precision"],

                "score_recall":
                    score_result["recall"],

                "score_f1":
                    score_result["f1"],

                "score_auc":
                    score_result["auc"],

                "score_shadow_val_auc":
                    score_result[
                        "shadow_val_auc"
                    ],

                "score_best_epoch":
                    score_result[
                        "best_epoch"
                    ],

                "temporal_accuracy":
                    temporal_result["accuracy"],

                "temporal_precision":
                    temporal_result["precision"],

                "temporal_recall":
                    temporal_result["recall"],

                "temporal_f1":
                    temporal_result["f1"],

                "temporal_auc":
                    temporal_result["auc"],

                "temporal_shadow_val_auc":
                    temporal_result[
                        "shadow_val_auc"
                    ],

                "temporal_best_epoch":
                    temporal_result[
                        "best_epoch"
                    ],

                "auc_gain":
                    gain,
            }
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    output_path = (
        RESULTS_ROOT
        / "mia_mlp_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\n\n" + "=" * 80
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        results_df[
            [
                "dataset",
                "score_auc",
                "temporal_auc",
                "auc_gain",
                "score_shadow_val_auc",
                "temporal_shadow_val_auc",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        output_path
    )

    print(
        "=" * 80
    )