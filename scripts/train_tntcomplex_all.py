#!/usr/bin/env python3
# ============================================================
# train_tntcomplex_all.py
#
# Controlled PyTorch implementation of TNTComplEx for the
# existing privacy-split TKG pipeline.
#
# Trains 8 models:
#   4 datasets x {victim, shadow}
#
# Training data:
#   victim -> victim_train.txt only
#   shadow -> shadow_train.txt only
#
# NEVER uses victim_test/shadow_test for TKG training.
#
# Checkpoints:
#   models/<DATASET>/tntcomplex_victim.pt
#   models/<DATASET>/tntcomplex_shadow.pt
#
# TNTComplEx score:
#
#   q(h,r,t) = h ⊙ ( r_temporal ⊙ time + r_static )
#
#   score(h,r,o,t)
#     = Re( < h,
#             r_temporal ⊙ time + r_static,
#             conj(o) > )
#
# This corresponds to a temporal ComplEx component plus a
# non-temporal ComplEx component.
#
# Training:
#   - reciprocal relations
#   - full-softmax CrossEntropyLoss
#   - N3-style embedding regularization
#   - temporal smoothness regularization
#
# NOTE:
# This is a controlled PyTorch implementation aligned with the
# TNTComplEx formulation, not a claim of bit-for-bit reproduction
# of the authors' original repository/hyperparameter tuning.
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

BASE_DIR = Path("/home/harrouch/Paper2/Paper2")
PROCESSED_ROOT = BASE_DIR / "processed"
MODEL_ROOT = BASE_DIR / "models"

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

SEED = 42

EMBEDDING_DIM = 200
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-3

# Keep these aligned with the prior controlled TComplEx setup.
N3_WEIGHT = 1e-3
TIME_SMOOTH_WEIGHT = 1e-3

NUM_WORKERS = 0
PIN_MEMORY = True

MODEL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA LOADING
# ============================================================

def read_facts(path):
    """
    Expected processed format:
        h r o t

    Whitespace/tab separated integer IDs.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    facts = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 4:
                raise ValueError(
                    f"{path}:{line_no}: "
                    f"expected >=4 columns, got {parts}"
                )

            h = int(parts[0])
            r = int(parts[1])
            o = int(parts[2])
            t = int(parts[3])

            facts.append(
                (
                    h,
                    r,
                    o,
                    t,
                )
            )

    if not facts:
        raise ValueError(
            f"No facts found in {path}"
        )

    return facts


def load_time2id(dataset_name):
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

    return {
        str(k): int(v)
        for k, v in raw.items()
    }


def compute_global_dimensions(dataset_name):
    """
    Same idea as the previous TeRo/TComplEx pipeline:
    compute vocabulary dimensions across all four privacy splits,
    so victim and shadow models use identical ID spaces.

    IMPORTANT:
    only the role's *_train split is later used for optimization.
    """

    split_names = [
        "victim_train",
        "victim_test",
        "shadow_train",
        "shadow_test",
    ]

    max_entity = -1
    max_relation = -1
    max_time = -1
    total = 0

    dataset_dir = (
        PROCESSED_ROOT
        /
        dataset_name
    )

    for split_name in split_names:

        path = (
            dataset_dir
            /
            f"{split_name}.txt"
        )

        facts = read_facts(path)

        total += len(facts)

        for h, r, o, t in facts:
            max_entity = max(
                max_entity,
                h,
                o,
            )

            max_relation = max(
                max_relation,
                r,
            )

            max_time = max(
                max_time,
                t,
            )

    time2id = load_time2id(
        dataset_name
    )

    # Be safe if time2id includes timestamp IDs not present in facts.
    if time2id:
        max_time = max(
            max_time,
            max(time2id.values()),
        )

    return {
        "num_entities":
            max_entity + 1,

        "num_relations":
            max_relation + 1,

        "num_times":
            max_time + 1,

        "all_processed_facts":
            total,
    }


# ============================================================
# DATASET
# ============================================================

class ReciprocalQueryDataset(Dataset):
    """
    From each original fact:
        (h, r, o, t)

    creates two object-prediction queries:

        (h, r, t)       -> o
        (o, r + R, t)   -> h

    where R = number of base relations.
    """

    def __init__(
        self,
        facts,
        num_base_relations,
    ):
        self.queries = []

        for h, r, o, t in facts:

            self.queries.append(
                (
                    h,
                    r,
                    t,
                    o,
                )
            )

            self.queries.append(
                (
                    o,
                    r + num_base_relations,
                    t,
                    h,
                )
            )

    def __len__(self):
        return len(
            self.queries
        )

    def __getitem__(self, idx):
        h, r, t, target = (
            self.queries[idx]
        )

        return (
            torch.tensor(
                h,
                dtype=torch.long,
            ),
            torch.tensor(
                r,
                dtype=torch.long,
            ),
            torch.tensor(
                t,
                dtype=torch.long,
            ),
            torch.tensor(
                target,
                dtype=torch.long,
            ),
        )


# ============================================================
# COMPLEX HELPERS
# ============================================================

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

    return (
        out_re,
        out_im,
    )


def complex_abs(
    re,
    im,
):
    return torch.sqrt(
        re.pow(2)
        +
        im.pow(2)
        +
        1e-12
    )


# ============================================================
# TNTComplEx
# ============================================================

class TNTComplEx(nn.Module):
    """
    TNTComplEx with:
      - complex entity embeddings
      - complex temporal relation embeddings
      - complex static relation embeddings
      - complex timestamp embeddings

    For query (h,r,?,t):

        rel_dynamic = r_temporal ⊙ time
        rel_total   = rel_dynamic + r_static
        q           = h ⊙ rel_total

    and

        score(h,r,o,t)
          = Re(q * conj(o))

    which gives:
        q_re * o_re + q_im * o_im
    summed across dimensions.
    """

    def __init__(
        self,
        num_entities,
        num_relations_with_reciprocals,
        num_times,
        embedding_dim,
    ):
        super().__init__()

        self.num_entities = (
            num_entities
        )

        self.num_relations = (
            num_relations_with_reciprocals
        )

        self.num_times = (
            num_times
        )

        self.embedding_dim = (
            embedding_dim
        )

        # Entities
        self.ent_re = nn.Embedding(
            num_entities,
            embedding_dim,
        )

        self.ent_im = nn.Embedding(
            num_entities,
            embedding_dim,
        )

        # Temporal relation component v^t_r
        self.rel_time_re = nn.Embedding(
            num_relations_with_reciprocals,
            embedding_dim,
        )

        self.rel_time_im = nn.Embedding(
            num_relations_with_reciprocals,
            embedding_dim,
        )

        # Static relation component v_r
        self.rel_static_re = nn.Embedding(
            num_relations_with_reciprocals,
            embedding_dim,
        )

        self.rel_static_im = nn.Embedding(
            num_relations_with_reciprocals,
            embedding_dim,
        )

        # Timestamp embeddings
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
        """
        Small random initialization suitable for complex KGE models.
        """

        scale = (
            1.0
            /
            np.sqrt(
                self.embedding_dim
            )
        )

        embeddings = [
            self.ent_re,
            self.ent_im,
            self.rel_time_re,
            self.rel_time_im,
            self.rel_static_re,
            self.rel_static_im,
            self.time_re,
            self.time_im,
        ]

        for emb in embeddings:
            nn.init.uniform_(
                emb.weight,
                a=-scale,
                b=scale,
            )

    def query_embedding(
        self,
        h,
        r,
        t,
    ):
        h_re = self.ent_re(h)
        h_im = self.ent_im(h)

        rt_re = self.rel_time_re(r)
        rt_im = self.rel_time_im(r)

        rs_re = self.rel_static_re(r)
        rs_im = self.rel_static_im(r)

        time_re = self.time_re(t)
        time_im = self.time_im(t)

        # Temporal relation:
        # r_temporal ⊙ time
        dyn_re, dyn_im = complex_mul(
            rt_re,
            rt_im,
            time_re,
            time_im,
        )

        # TNT component:
        # r_temporal ⊙ time + r_static
        rel_re = (
            dyn_re
            +
            rs_re
        )

        rel_im = (
            dyn_im
            +
            rs_im
        )

        # h ⊙ effective_relation
        q_re, q_im = complex_mul(
            h_re,
            h_im,
            rel_re,
            rel_im,
        )

        return (
            q_re,
            q_im,
            h_re,
            h_im,
            rt_re,
            rt_im,
            rs_re,
            rs_im,
            time_re,
            time_im,
            rel_re,
            rel_im,
        )

    def forward(
        self,
        h,
        r,
        t,
    ):
        """
        Full-softmax logits over ALL candidate objects.

        score(q,o) =
            q_re @ ent_re[o]
          + q_im @ ent_im[o]
        """

        (
            q_re,
            q_im,
            *_,
        ) = self.query_embedding(
            h,
            r,
            t,
        )

        logits = (
            q_re
            @
            self.ent_re.weight.T
            +
            q_im
            @
            self.ent_im.weight.T
        )

        return logits

    def score(
        self,
        h,
        r,
        o,
        t,
    ):
        """
        Individual quadruple score.
        Useful later for MIA / T-MIA.
        """

        (
            q_re,
            q_im,
            *_,
        ) = self.query_embedding(
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

    def regularization(
        self,
        h,
        r,
        t,
        target,
    ):
        """
        Controlled N3-style regularizer.

        This is intentionally kept structurally similar to the
        prior TComplEx experiment for a fair controlled comparison.
        """

        h_re = self.ent_re(h)
        h_im = self.ent_im(h)

        o_re = self.ent_re(target)
        o_im = self.ent_im(target)

        rt_re = self.rel_time_re(r)
        rt_im = self.rel_time_im(r)

        rs_re = self.rel_static_re(r)
        rs_im = self.rel_static_im(r)

        time_re = self.time_re(t)
        time_im = self.time_im(t)

        h_abs = complex_abs(
            h_re,
            h_im,
        )

        o_abs = complex_abs(
            o_re,
            o_im,
        )

        rt_abs = complex_abs(
            rt_re,
            rt_im,
        )

        rs_abs = complex_abs(
            rs_re,
            rs_im,
        )

        time_abs = complex_abs(
            time_re,
            time_im,
        )

        n3 = (
            h_abs.pow(3).mean()
            +
            o_abs.pow(3).mean()
            +
            rt_abs.pow(3).mean()
            +
            rs_abs.pow(3).mean()
            +
            time_abs.pow(3).mean()
        )

        return n3

    def temporal_smoothness(
        self,
    ):
        """
        Penalize changes between consecutive timestamp embeddings.
        Processed time IDs are globally chronological in the
        existing pipeline.
        """

        if self.num_times <= 1:
            return torch.tensor(
                0.0,
                device=
                    self.time_re.weight.device,
            )

        diff_re = (
            self.time_re.weight[1:]
            -
            self.time_re.weight[:-1]
        )

        diff_im = (
            self.time_im.weight[1:]
            -
            self.time_im.weight[:-1]
        )

        diff_abs = complex_abs(
            diff_re,
            diff_im,
        )

        return diff_abs.pow(3).mean()


# ============================================================
# TRAINING
# ============================================================

def train_one(
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
        f"TNTComplEx | "
        f"{dataset_name} | "
        f"{role.upper()}"
    )

    print(
        "=" * 100
    )

    dimensions = (
        compute_global_dimensions(
            dataset_name
        )
    )

    num_entities = (
        dimensions[
            "num_entities"
        ]
    )

    num_base_relations = (
        dimensions[
            "num_relations"
        ]
    )

    num_times = (
        dimensions[
            "num_times"
        ]
    )

    print(
        f"Global entities : "
        f"{num_entities}"
    )

    print(
        f"Base relations  : "
        f"{num_base_relations}"
    )

    print(
        f"Times           : "
        f"{num_times}"
    )

    print(
        f"All proc. facts : "
        f"{dimensions['all_processed_facts']}"
    )

    train_path = (
        PROCESSED_ROOT
        /
        dataset_name
        /
        f"{role}_train.txt"
    )

    train_facts = read_facts(
        train_path
    )

    print(
        f"Training facts  : "
        f"{len(train_facts)}"
    )

    train_dataset = (
        ReciprocalQueryDataset(
            facts=
                train_facts,
            num_base_relations=
                num_base_relations,
        )
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=
            BATCH_SIZE,
        shuffle=True,
        num_workers=
            NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and
            device.type == "cuda"
        ),
        generator=
            generator,
        drop_last=False,
    )

    # Reciprocal relations double the relation vocabulary.
    num_relations_with_reciprocals = (
        2
        *
        num_base_relations
    )

    model = TNTComplEx(
        num_entities=
            num_entities,
        num_relations_with_reciprocals=
            num_relations_with_reciprocals,
        num_times=
            num_times,
        embedding_dim=
            EMBEDDING_DIM,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=
            LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    print(
        f"Queries/epoch   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Batch size      : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs          : "
        f"{EPOCHS}"
    )

    print(
        f"Embedding dim   : "
        f"{EMBEDDING_DIM}"
    )

    print(
        f"LR              : "
        f"{LEARNING_RATE}"
    )

    print(
        f"N3 weight       : "
        f"{N3_WEIGHT}"
    )

    print(
        f"Time smooth     : "
        f"{TIME_SMOOTH_WEIGHT}"
    )

    start_time = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0
        total_ce = 0.0
        total_n3 = 0.0
        total_smooth = 0.0
        total_examples = 0

        for (
            h,
            r,
            t,
            target,
        ) in train_loader:

            h = h.to(
                device,
                non_blocking=True,
            )

            r = r.to(
                device,
                non_blocking=True,
            )

            t = t.to(
                device,
                non_blocking=True,
            )

            target = target.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                h,
                r,
                t,
            )

            ce_loss = criterion(
                logits,
                target,
            )

            n3_loss = (
                model.regularization(
                    h=
                        h,
                    r=
                        r,
                    t=
                        t,
                    target=
                        target,
                )
            )

            smooth_loss = (
                model.temporal_smoothness()
            )

            loss = (
                ce_loss
                +
                N3_WEIGHT
                *
                n3_loss
                +
                TIME_SMOOTH_WEIGHT
                *
                smooth_loss
            )

            loss.backward()

            optimizer.step()

            batch_n = h.shape[0]

            total_examples += (
                batch_n
            )

            total_loss += (
                float(
                    loss.detach()
                )
                *
                batch_n
            )

            total_ce += (
                float(
                    ce_loss.detach()
                )
                *
                batch_n
            )

            total_n3 += (
                float(
                    n3_loss.detach()
                )
                *
                batch_n
            )

            total_smooth += (
                float(
                    smooth_loss.detach()
                )
                *
                batch_n
            )

        avg_loss = (
            total_loss
            /
            total_examples
        )

        avg_ce = (
            total_ce
            /
            total_examples
        )

        avg_n3 = (
            total_n3
            /
            total_examples
        )

        avg_smooth = (
            total_smooth
            /
            total_examples
        )

        if (
            epoch == 1
            or
            epoch % 10 == 0
            or
            epoch == EPOCHS
        ):

            elapsed = (
                time.time()
                -
                start_time
            )

            print(
                f"Epoch "
                f"{epoch:03d}/{EPOCHS} | "
                f"loss={avg_loss:.6f} | "
                f"CE={avg_ce:.6f} | "
                f"N3={avg_n3:.6f} | "
                f"smooth={avg_smooth:.6f} | "
                f"elapsed={elapsed/60.0:.1f} min"
            )

    # ========================================================
    # SAVE
    # ========================================================

    output_dir = (
        MODEL_ROOT
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
        f"tntcomplex_{role}.pt"
    )

    checkpoint = {
        "model_name":
            "TNTComplEx",

        "implementation_note":
            (
                "Controlled PyTorch TNTComplEx implementation "
                "for cross-architecture privacy experiments; "
                "not claimed as bit-for-bit reproduction of "
                "the original authors' code."
            ),

        "model_state_dict":
            model.state_dict(),

        "dataset":
            dataset_name,

        "role":
            role,

        "num_entities":
            num_entities,

        "num_base_relations":
            num_base_relations,

        "num_relations_with_reciprocals":
            num_relations_with_reciprocals,

        "num_times":
            num_times,

        "embedding_dim":
            EMBEDDING_DIM,

        "training_config": {
            "seed":
                SEED,

            "batch_size":
                BATCH_SIZE,

            "epochs":
                EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "n3_weight":
                N3_WEIGHT,

            "time_smooth_weight":
                TIME_SMOOTH_WEIGHT,

            "loss":
                "full-softmax cross entropy",

            "reciprocal_relations":
                True,

            "training_split":
                f"{role}_train only",
        },

        "time2id":
            load_time2id(
                dataset_name
            ),
    }

    torch.save(
        checkpoint,
        output_path,
    )

    print(
        f"\nSaved: {output_path}"
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    print(
        "=" * 100
    )

    print(
        "TRAIN ALL TNTComplEx "
        "VICTIM / SHADOW MODELS"
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

    print(
        "\nDatasets:"
    )

    for dataset_name in DATASETS:
        print(
            f"  - {dataset_name}"
        )

    for dataset_name in DATASETS:

        for role in ROLES:

            set_seed(SEED)

            train_one(
                dataset_name=
                    dataset_name,
                role=
                    role,
                device=
                    device,
            )

    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "ALL TNTComplEx MODELS "
        "TRAINED SUCCESSFULLY"
    )

    print(
        "=" * 100
    )
