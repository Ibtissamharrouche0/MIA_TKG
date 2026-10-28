#!/usr/bin/env python3
# ============================================================
# train_tcomplex_all.py
#
# Train TComplEx models on the SAME privacy splits used for TeRo:
#   - victim model  -> victim_train.txt ONLY
#   - shadow model  -> shadow_train.txt ONLY
#
# Datasets:
#   ICEWS05-15, ICEWS14, YAGO11k, WIKIDATA12k
#
# TComplEx score:
#   Re(< e_h, r, conj(e_o), tau_t >)
#
# Training:
#   - reciprocal relations
#   - full-softmax cross-entropy over candidate objects
#   - N3-style embedding regularization
#   - temporal smoothness regularization
#
# IMPORTANT:
#   victim_test.txt and shadow_test.txt are NOT used to train TComplEx.
#   They remain non-member sets for the later MIA/T-MIA experiment.
# ============================================================

from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

SEED = 42

DATASETS = [
    "ICEWS05-15",
    "ICEWS14",
    "YAGO11k",
    "WIKIDATA12k",
]

ROLES = [
    "victim",
    "shadow",
]

# Explorer project
BASE_DIR = Path("/home/harrouch/Paper2/Paper2")
PROCESSED_ROOT = BASE_DIR / "processed"
MODEL_ROOT = BASE_DIR / "models"

# Fixed setup for a controlled cross-model privacy comparison.
EMBEDDING_DIM = 200
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

# Regularization.
# These are fixed across datasets so that we do not tune on victim_test/shadow_test.
N3_WEIGHT = 1e-3
TIME_SMOOTH_WEIGHT = 1e-3

PRINT_EVERY = 10


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
# DATA I/O
# ============================================================

def load_facts(path):
    facts = []

    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 4:
                raise ValueError(
                    f"{path}, line {line_number}: "
                    f"expected 4 integer columns, got {len(parts)}"
                )

            h, r, o, t = map(int, parts)
            facts.append((h, r, o, t))

    return facts


def load_time2id(dataset_dir):
    path = dataset_dir / "time2id.json"

    with open(path, "r") as f:
        return json.load(f)


def get_global_dimensions(dataset_dir):
    """
    Compute dimensions over all four processed privacy partitions.
    This keeps victim and shadow models in exactly the same vocabulary.
    """
    split_files = [
        "victim_train.txt",
        "victim_test.txt",
        "shadow_train.txt",
        "shadow_test.txt",
    ]

    max_entity = -1
    max_relation = -1
    max_time = -1
    total_facts = 0

    for filename in split_files:
        path = dataset_dir / filename
        facts = load_facts(path)
        total_facts += len(facts)

        for h, r, o, t in facts:
            max_entity = max(max_entity, h, o)
            max_relation = max(max_relation, r)
            max_time = max(max_time, t)

    return (
        max_entity + 1,
        max_relation + 1,
        max_time + 1,
        total_facts,
    )


# ============================================================
# DATASET
# ============================================================

class TKGDataset(Dataset):
    def __init__(self, facts):
        self.facts = torch.tensor(
            facts,
            dtype=torch.long,
        )

    def __len__(self):
        return self.facts.shape[0]

    def __getitem__(self, idx):
        return self.facts[idx]


# ============================================================
# TCOMPLEX
# ============================================================

class TComplEx(nn.Module):
    """
    TComplEx:
        score(h,r,o,t) = Re(<h, r, conj(o), time>)

    We allocate 2 * num_relations relation embeddings because training
    uses reciprocal relations:
        (h, r, o, t)
        (o, r + num_relations, h, t)
    """

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

        self.reset_parameters()

    def reset_parameters(self):
        # Stable initialization for complex factorization.
        scale = 1.0 / np.sqrt(self.embedding_dim)

        for emb in [
            self.ent_re,
            self.ent_im,
            self.rel_re,
            self.rel_im,
            self.time_re,
            self.time_im,
        ]:
            nn.init.normal_(
                emb.weight,
                mean=0.0,
                std=scale,
            )

    @staticmethod
    def complex_mul(a_re, a_im, b_re, b_im):
        out_re = a_re * b_re - a_im * b_im
        out_im = a_re * b_im + a_im * b_re
        return out_re, out_im

    def query_embedding(self, h, r, t):
        """
        q = e_h ⊙ r ⊙ tau_t
        """
        h_re = self.ent_re(h)
        h_im = self.ent_im(h)

        r_re = self.rel_re(r)
        r_im = self.rel_im(r)

        t_re = self.time_re(t)
        t_im = self.time_im(t)

        q_re, q_im = self.complex_mul(
            h_re, h_im,
            r_re, r_im,
        )

        q_re, q_im = self.complex_mul(
            q_re, q_im,
            t_re, t_im,
        )

        return q_re, q_im

    def score(self, h, r, o, t):
        """
        Score individual quadruples.

        Re(q * conj(o)) = q_re*o_re + q_im*o_im
        """
        q_re, q_im = self.query_embedding(h, r, t)

        o_re = self.ent_re(o)
        o_im = self.ent_im(o)

        return torch.sum(
            q_re * o_re + q_im * o_im,
            dim=1,
        )

    def score_all_objects(self, h, r, t):
        """
        Scores every entity as the candidate object.
        Shape: [batch, num_entities]
        """
        q_re, q_im = self.query_embedding(h, r, t)

        return (
            q_re @ self.ent_re.weight.t()
            +
            q_im @ self.ent_im.weight.t()
        )

    def n3_regularizer(self, h, r, o, t):
        """
        N3-style regularization based on complex magnitudes.
        """
        eps = 1e-12

        h_mag = torch.sqrt(
            self.ent_re(h).pow(2)
            + self.ent_im(h).pow(2)
            + eps
        )

        r_mag = torch.sqrt(
            self.rel_re(r).pow(2)
            + self.rel_im(r).pow(2)
            + eps
        )

        o_mag = torch.sqrt(
            self.ent_re(o).pow(2)
            + self.ent_im(o).pow(2)
            + eps
        )

        t_mag = torch.sqrt(
            self.time_re(t).pow(2)
            + self.time_im(t).pow(2)
            + eps
        )

        return (
            h_mag.pow(3).mean()
            + r_mag.pow(3).mean()
            + o_mag.pow(3).mean()
            + t_mag.pow(3).mean()
        )

    def temporal_smoothness(self):
        """
        Penalize abrupt changes between consecutive timestamp embeddings.
        """
        if self.num_times <= 1:
            return torch.zeros(
                (),
                device=self.time_re.weight.device,
            )

        d_re = (
            self.time_re.weight[1:]
            - self.time_re.weight[:-1]
        )
        d_im = (
            self.time_im.weight[1:]
            - self.time_im.weight[:-1]
        )

        diff_mag = torch.sqrt(
            d_re.pow(2)
            + d_im.pow(2)
            + 1e-12
        )

        return diff_mag.pow(3).mean()


# ============================================================
# TRAINING
# ============================================================

def train_one_model(
    dataset_name,
    role,
    device,
):
    dataset_dir = (
        PROCESSED_ROOT
        / dataset_name
    )

    output_dir = (
        MODEL_ROOT
        / dataset_name
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        dataset_dir
        / f"{role}_train.txt"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing training split: {train_path}"
        )

    facts = load_facts(train_path)

    (
        num_entities,
        num_relations,
        num_times,
        total_processed_facts,
    ) = get_global_dimensions(
        dataset_dir
    )

    time2id = load_time2id(
        dataset_dir
    )

    print("\n" + "=" * 80)
    print(
        f"{dataset_name} | {role.upper()} | TComplEx"
    )
    print("=" * 80)

    print(f"Training file       : {train_path}")
    print(f"Training facts      : {len(facts)}")
    print(f"All processed facts : {total_processed_facts}")
    print(f"Entities            : {num_entities}")
    print(f"Base relations      : {num_relations}")
    print(f"Relations w/ inverse: {2 * num_relations}")
    print(f"Timestamps          : {num_times}")
    print(f"Embedding dim       : {EMBEDDING_DIM}")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Epochs              : {EPOCHS}")
    print(f"Learning rate       : {LEARNING_RATE}")
    print(f"N3 weight           : {N3_WEIGHT}")
    print(f"Time smooth weight  : {TIME_SMOOTH_WEIGHT}")
    print(f"Device              : {device}")

    train_dataset = TKGDataset(
        facts
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = TComplEx(
        num_entities=num_entities,
        num_relations=num_relations,
        num_times=num_times,
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    start_time = time.perf_counter()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        total_loss = 0.0
        total_ce = 0.0
        total_reg = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(
                device,
                non_blocking=True,
            )

            h = batch[:, 0]
            r = batch[:, 1]
            o = batch[:, 2]
            t = batch[:, 3]

            # --------------------------------------------
            # Forward query:
            #   (h, r, ?, t) -> o
            # --------------------------------------------
            h_f = h
            r_f = r
            target_f = o
            t_f = t

            # --------------------------------------------
            # Reciprocal query:
            #   (o, r_inverse, ?, t) -> h
            # --------------------------------------------
            h_b = o
            r_b = (
                r
                + num_relations
            )
            target_b = h
            t_b = t

            # Combine both directions.
            q_h = torch.cat(
                [h_f, h_b],
                dim=0,
            )
            q_r = torch.cat(
                [r_f, r_b],
                dim=0,
            )
            q_t = torch.cat(
                [t_f, t_b],
                dim=0,
            )
            targets = torch.cat(
                [target_f, target_b],
                dim=0,
            )

            logits = model.score_all_objects(
                q_h,
                q_r,
                q_t,
            )

            ce_loss = criterion(
                logits,
                targets,
            )

            # For regularization, targets are the object
            # embeddings for the combined reciprocal batch.
            n3 = model.n3_regularizer(
                q_h,
                q_r,
                targets,
                q_t,
            )

            smooth = model.temporal_smoothness()

            loss = (
                ce_loss
                + N3_WEIGHT * n3
                + TIME_SMOOTH_WEIGHT * smooth
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_reg += (
                N3_WEIGHT * n3.item()
                + TIME_SMOOTH_WEIGHT * smooth.item()
            )
            num_batches += 1

        avg_loss = (
            total_loss
            / max(num_batches, 1)
        )
        avg_ce = (
            total_ce
            / max(num_batches, 1)
        )
        avg_reg = (
            total_reg
            / max(num_batches, 1)
        )

        if (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == EPOCHS
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"loss={avg_loss:.6f} | "
                f"CE={avg_ce:.6f} | "
                f"reg={avg_reg:.6f} | "
                f"time={elapsed/60:.1f} min"
            )

    total_seconds = (
        time.perf_counter()
        - start_time
    )

    checkpoint_path = (
        output_dir
        / f"tcomplex_{role}.pt"
    )

    checkpoint = {
        "dataset": dataset_name,
        "role": role,
        "model_type": "TComplEx",
        "model_state_dict": model.state_dict(),

        "num_entities": num_entities,
        "num_relations": num_relations,
        "total_num_relations": 2 * num_relations,
        "num_times": num_times,
        "embedding_dim": EMBEDDING_DIM,

        "uses_reciprocal_relations": True,
        "time2id": time2id,

        "training_config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "n3_weight": N3_WEIGHT,
            "time_smooth_weight": TIME_SMOOTH_WEIGHT,
            "training_facts": len(facts),
            "loss": (
                "full-softmax cross-entropy "
                "+ N3-style regularization "
                "+ temporal smoothness"
            ),
        },
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(
        f"\nSaved: {checkpoint_path}"
    )
    print(
        f"Total training time: "
        f"{total_seconds/3600:.2f} h"
    )

    return checkpoint_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("TCOMPLEX TRAINING — VICTIM + SHADOW — ALL DATASETS")
    print("=" * 80)
    print(f"Device   : {device}")

    if torch.cuda.is_available():
        print(
            "GPU      : "
            + torch.cuda.get_device_name(0)
        )

    print(f"Datasets : {DATASETS}")
    print(f"Roles    : {ROLES}")

    successful = []
    failed = []

    all_start = time.perf_counter()

    for dataset_name in DATASETS:
        for role in ROLES:
            try:
                path = train_one_model(
                    dataset_name=dataset_name,
                    role=role,
                    device=device,
                )

                successful.append(
                    (
                        dataset_name,
                        role,
                        str(path),
                    )
                )

                # Reduce fragmentation between large runs.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(
                    f"\nERROR | "
                    f"{dataset_name} | "
                    f"{role}: "
                    f"{type(e).__name__}: {e}"
                )

                failed.append(
                    (
                        dataset_name,
                        role,
                        str(e),
                    )
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    all_seconds = (
        time.perf_counter()
        - all_start
    )

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)

    print("\nSuccessful:")
    if successful:
        for d, role, path in successful:
            print(
                f"  ✓ {d:14s} | "
                f"{role:6s} | {path}"
            )
    else:
        print("  None")

    print("\nFailed:")
    if failed:
        for d, role, err in failed:
            print(
                f"  ✗ {d:14s} | "
                f"{role:6s} | {err}"
            )
    else:
        print("  None")

    print(
        f"\nTotal wall-clock time: "
        f"{all_seconds/3600:.2f} h"
    )
