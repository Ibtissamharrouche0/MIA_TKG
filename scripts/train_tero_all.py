import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

SEED = 42

PROCESSED_ROOT = Path(
    "/home/harrouche/Paper2/processed"
)

MODEL_ROOT = Path(
    "/home/harrouche/Paper2/models"
)

MODEL_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


DATASETS = [
    "ICEWS05-15",
    "ICEWS14",
    "YAGO11k",
    "WIKIDATA12k",
]


TRAIN_SPLITS = {
    "victim": "victim_train.txt",
    "shadow": "shadow_train.txt",
}


# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

EMBEDDING_DIM = 200
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 0.001
MARGIN = 1.0
NUM_WORKERS = 0


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
# LOAD FACTS
# ============================================================

def load_facts(path):

    facts = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 4:

                raise ValueError(
                    f"Expected 4 columns in {path}, "
                    f"line {line_number}: {line}"
                )

            h = int(parts[0])
            r = int(parts[1])
            o = int(parts[2])
            t = int(parts[3])

            facts.append(
                (h, r, o, t)
            )

    return facts


# ============================================================
# GET GLOBAL DATASET DIMENSIONS
# ============================================================

def get_global_dimensions(dataset_dir):

    """
    Important:
    dimensions are calculated over ALL 4 splits,
    not only victim_train or shadow_train.

    This ensures both victim and shadow models use the same:
        - entity space
        - relation space
        - temporal space
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

        if not path.exists():

            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        facts = load_facts(path)

        total_facts += len(facts)

        for h, r, o, t in facts:

            max_entity = max(
                max_entity,
                h,
                o
            )

            max_relation = max(
                max_relation,
                r
            )

            max_time = max(
                max_time,
                t
            )

    num_entities = max_entity + 1
    num_relations = max_relation + 1
    num_times = max_time + 1

    return (
        num_entities,
        num_relations,
        num_times,
        total_facts
    )


# ============================================================
# DATASET CLASS
# ============================================================

class TKGDataset(Dataset):

    def __init__(
        self,
        facts,
        num_entities
    ):

        self.facts = torch.tensor(
            facts,
            dtype=torch.long
        )

        self.num_entities = num_entities

    def __len__(self):

        return len(self.facts)

    def __getitem__(self, idx):

        positive = self.facts[idx]

        h = positive[0]
        r = positive[1]
        o = positive[2]
        t = positive[3]

        # ----------------------------------------------------
        # Negative tail sampling
        # ----------------------------------------------------

        negative_o = torch.randint(
            low=0,
            high=self.num_entities,
            size=(1,),
            dtype=torch.long
        )[0]

        # Ensure negative tail differs from true tail
        while negative_o == o:

            negative_o = torch.randint(
                low=0,
                high=self.num_entities,
                size=(1,),
                dtype=torch.long
            )[0]

        negative = torch.stack(
            [
                h,
                r,
                negative_o,
                t
            ]
        )

        return positive, negative


# ============================================================
# TeRo-LIKE MODEL
# ============================================================

class TeRo(nn.Module):

    def __init__(
        self,
        num_entities,
        num_relations,
        num_times,
        embedding_dim
    ):

        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_times = num_times
        self.embedding_dim = embedding_dim

        # ----------------------------------------------------
        # Complex entity embeddings
        # ----------------------------------------------------

        self.ent_re = nn.Embedding(
            num_entities,
            embedding_dim
        )

        self.ent_im = nn.Embedding(
            num_entities,
            embedding_dim
        )

        # ----------------------------------------------------
        # Complex relation embeddings
        # ----------------------------------------------------

        self.rel_re = nn.Embedding(
            num_relations,
            embedding_dim
        )

        self.rel_im = nn.Embedding(
            num_relations,
            embedding_dim
        )

        # ----------------------------------------------------
        # Temporal phase embedding
        # ----------------------------------------------------

        self.time_emb = nn.Embedding(
            num_times,
            embedding_dim
        )

        self.reset_parameters()

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    def reset_parameters(self):

        nn.init.xavier_uniform_(
            self.ent_re.weight
        )

        nn.init.xavier_uniform_(
            self.ent_im.weight
        )

        nn.init.xavier_uniform_(
            self.rel_re.weight
        )

        nn.init.xavier_uniform_(
            self.rel_im.weight
        )

        nn.init.uniform_(
            self.time_emb.weight,
            a=-np.pi,
            b=np.pi
        )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    def score(
        self,
        h,
        r,
        o,
        t
    ):

        h_re = self.ent_re(h)
        h_im = self.ent_im(h)

        o_re = self.ent_re(o)
        o_im = self.ent_im(o)

        r_re = self.rel_re(r)
        r_im = self.rel_im(r)

        phase = self.time_emb(t)

        cos_t = torch.cos(
            phase
        )

        sin_t = torch.sin(
            phase
        )

        # ----------------------------------------------------
        # Rotate head in temporal space
        # ----------------------------------------------------

        h_t_re = (
            h_re * cos_t
            -
            h_im * sin_t
        )

        h_t_im = (
            h_re * sin_t
            +
            h_im * cos_t
        )

        # ----------------------------------------------------
        # Rotate object
        # ----------------------------------------------------

        o_t_re = (
            o_re * cos_t
            -
            o_im * sin_t
        )

        o_t_im = (
            o_re * sin_t
            +
            o_im * cos_t
        )

        # ----------------------------------------------------
        # Relation transformation
        # ----------------------------------------------------

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
                diff_re ** 2
                +
                diff_im ** 2,
                dim=1
            )
            +
            1e-12
        )

        # Higher = more plausible
        score = -distance

        return score

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        facts
    ):

        h = facts[:, 0]
        r = facts[:, 1]
        o = facts[:, 2]
        t = facts[:, 3]

        return self.score(
            h,
            r,
            o,
            t
        )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    dataset_name,
    role,
    train_path,
    num_entities,
    num_relations,
    num_times,
    time2id,
    device
):

    print("\n" + "=" * 75)

    print(
        f"TRAINING {dataset_name} — {role.upper()}"
    )

    print("=" * 75)

    facts = load_facts(
        train_path
    )

    print(
        f"Training facts : {len(facts)}"
    )

    print(
        f"Entities       : {num_entities}"
    )

    print(
        f"Relations      : {num_relations}"
    )

    print(
        f"Timestamps     : {num_times}"
    )

    print(
        f"Embedding dim  : {EMBEDDING_DIM}"
    )

    print(
        f"Device         : {device}"
    )

    dataset = TKGDataset(
        facts=facts,
        num_entities=num_entities
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    model = TeRo(
        num_entities=num_entities,
        num_relations=num_relations,
        num_times=num_times,
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MarginRankingLoss(
        margin=MARGIN
    )

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        total_loss = 0.0
        total_batches = 0

        for (
            positive,
            negative
        ) in loader:

            positive = positive.to(
                device
            )

            negative = negative.to(
                device
            )

            optimizer.zero_grad()

            positive_score = model(
                positive
            )

            negative_score = model(
                negative
            )

            target = torch.ones_like(
                positive_score
            )

            loss = criterion(
                positive_score,
                negative_score,
                target
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            total_batches += 1

        avg_loss = (
            total_loss
            /
            max(total_batches, 1)
        )

        if (
            epoch == 1
            or
            epoch % 10 == 0
            or
            epoch == EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:03d}/{EPOCHS} "
                f"| Loss: "
                f"{avg_loss:.6f}"
            )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    output_dir = (
        MODEL_ROOT
        /
        dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        output_dir
        /
        f"tero_{role}.pt"
    )

    checkpoint = {
        "dataset": dataset_name,
        "role": role,

        "model_type": "TeRo-like",

        "model_state_dict":
            model.state_dict(),

        "num_entities":
            num_entities,

        "num_relations":
            num_relations,

        "num_times":
            num_times,

        "embedding_dim":
            EMBEDDING_DIM,

        "time2id":
            time2id,

        "training_config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "margin": MARGIN,
            "seed": SEED,
            "training_facts": len(facts)
        }
    }

    torch.save(
        checkpoint,
        checkpoint_path
    )

    print(
        f"\nSaved model:"
    )

    print(
        checkpoint_path
    )

    return checkpoint_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    set_seed(
        SEED
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print("=" * 75)
    print("TeRo VICTIM + SHADOW TRAINING")
    print("=" * 75)

    print(
        f"Device: {device}"
    )

    successful = []
    failed = []

    # ========================================================
    # DATASET LOOP
    # ========================================================

    for dataset_name in DATASETS:

        dataset_dir = (
            PROCESSED_ROOT
            /
            dataset_name
        )

        print("\n" + "#" * 75)

        print(
            f"DATASET: {dataset_name}"
        )

        print("#" * 75)

        try:

            # ------------------------------------------------
            # Load temporal mapping
            # ------------------------------------------------

            time2id_path = (
                dataset_dir
                /
                "time2id.json"
            )

            if not time2id_path.exists():

                raise FileNotFoundError(
                    f"Missing: {time2id_path}"
                )

            with open(
                time2id_path,
                "r",
                encoding="utf-8"
            ) as f:

                time2id = json.load(
                    f
                )

            # ------------------------------------------------
            # Determine GLOBAL dimensions
            # ------------------------------------------------

            (
                num_entities,
                num_relations,
                num_times,
                total_facts
            ) = get_global_dimensions(
                dataset_dir
            )

            print(
                f"Global entity space : "
                f"{num_entities}"
            )

            print(
                f"Global relation space: "
                f"{num_relations}"
            )

            print(
                f"Global time space    : "
                f"{num_times}"
            )

            print(
                f"All processed facts : "
                f"{total_facts}"
            )

            # ------------------------------------------------
            # Train victim and shadow
            # ------------------------------------------------

            for role, filename in TRAIN_SPLITS.items():

                train_path = (
                    dataset_dir
                    /
                    filename
                )

                if not train_path.exists():

                    raise FileNotFoundError(
                        f"Missing: {train_path}"
                    )

                checkpoint = train_model(
                    dataset_name=dataset_name,
                    role=role,
                    train_path=train_path,
                    num_entities=num_entities,
                    num_relations=num_relations,
                    num_times=num_times,
                    time2id=time2id,
                    device=device
                )

                successful.append(
                    f"{dataset_name}/{role}"
                )

        except Exception as e:

            print(
                f"\nERROR: {dataset_name}"
            )

            print(
                e
            )

            failed.append(
                dataset_name
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("TRAINING FINISHED")
    print("=" * 75)

    print("\nSuccessful models:")

    for name in successful:

        print(
            f"  ✓ {name}"
        )

    print("\nFailed:")

    if failed:

        for name in failed:

            print(
                f"  ✗ {name}"
            )

    else:

        print(
            "  None"
        )

    print(
        f"\nModels root:"
    )

    print(
        MODEL_ROOT
    )