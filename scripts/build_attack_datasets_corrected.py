from pathlib import Path
from datetime import datetime, timedelta
import json
import random

import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

PROCESSED_ROOT = Path("/home/harrouche/Paper2/processed")
MODEL_ROOT = Path("/home/harrouche/Paper2/models")
OUTPUT_ROOT = Path("/home/harrouche/Paper2/attack_data")

DATASETS = [
    "ICEWS05-15",
    "ICEWS14",
    "YAGO11k",
    "WIKIDATA12k",
]

ICEWS_DATASETS = {
    "ICEWS05-15",
    "ICEWS14",
}

YEAR_DATASETS = {
    "YAGO11k",
    "WIKIDATA12k",
}

BATCH_SIZE = 2048
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# TeRo MODEL
# Must be identical to the model used during training
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

        self.ent_re = nn.Embedding(
            num_entities,
            embedding_dim
        )

        self.ent_im = nn.Embedding(
            num_entities,
            embedding_dim
        )

        self.rel_re = nn.Embedding(
            num_relations,
            embedding_dim
        )

        self.rel_im = nn.Embedding(
            num_relations,
            embedding_dim
        )

        self.time_emb = nn.Embedding(
            num_times,
            embedding_dim
        )

    def score(self, h, r, o, t):

        h_re = self.ent_re(h)
        h_im = self.ent_im(h)

        o_re = self.ent_re(o)
        o_im = self.ent_im(o)

        r_re = self.rel_re(r)
        r_im = self.rel_im(r)

        phase = self.time_emb(t)

        cos_t = torch.cos(phase)
        sin_t = torch.sin(phase)

        # Temporal rotation of head
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

        # Temporal rotation of object
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

        pred_re = h_t_re + r_re
        pred_im = h_t_im + r_im

        diff_re = pred_re - o_t_re
        diff_im = pred_im - o_t_im

        distance = torch.sqrt(
            torch.sum(
                diff_re ** 2 + diff_im ** 2,
                dim=1
            )
            + 1e-12
        )

        return -distance


# ============================================================
# LOAD FACTS
# ============================================================

def load_facts(path):

    facts = []

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

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
# LOAD TEMPORAL MAPPING
# ============================================================

def load_time_mapping(dataset_name):

    path = (
        PROCESSED_ROOT
        /
        dataset_name
        /
        "time2id.json"
    )

    with open(path, "r", encoding="utf-8") as f:
        raw_time2id = json.load(f)

    # JSON keys are strings
    time2id = {
        str(time_value): int(time_id)
        for time_value, time_id
        in raw_time2id.items()
    }

    id2time = {
        int(time_id): str(time_value)
        for time_value, time_id
        in raw_time2id.items()
    }

    return time2id, id2time


# ============================================================
# GET REAL TEMPORAL NEIGHBOR
# ============================================================

def get_neighbor_time_id(
    dataset_name,
    original_time_id,
    delta,
    time2id,
    id2time
):

    # --------------------------------------------------------
    # Original timestamp
    # --------------------------------------------------------

    original_value = id2time[
        int(original_time_id)
    ]

    # ========================================================
    # ICEWS:
    # real day +/- delta
    # ========================================================

    if dataset_name in ICEWS_DATASETS:

        current_date = datetime.strptime(
            original_value,
            "%Y-%m-%d"
        )

        neighbor_date = (
            current_date
            +
            timedelta(days=delta)
        )

        neighbor_value = neighbor_date.strftime(
            "%Y-%m-%d"
        )

    # ========================================================
    # YAGO / WIKIDATA:
    # real year +/- delta
    # ========================================================

    elif dataset_name in YEAR_DATASETS:

        current_year = int(
            original_value
        )

        neighbor_year = (
            current_year
            +
            delta
        )

        neighbor_value = str(
            neighbor_year
        )

    else:

        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # If the real neighboring timestamp does not exist
    # in the temporal vocabulary, return None.
    #
    # We DO NOT use the next available ID.
    # --------------------------------------------------------

    return time2id.get(
        neighbor_value
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    dataset_name,
    model_type,
    device
):

    checkpoint_path = (
        MODEL_ROOT
        /
        dataset_name
        /
        f"tero_{model_type}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing model: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model = TeRo(
        num_entities=checkpoint["num_entities"],
        num_relations=checkpoint["num_relations"],
        num_times=checkpoint["num_times"],
        embedding_dim=checkpoint["embedding_dim"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# SCORE QUERIES
# ============================================================

def score_queries(
    model,
    queries,
    device
):

    scores = []

    with torch.no_grad():

        for start in range(
            0,
            len(queries),
            BATCH_SIZE
        ):

            batch = queries[
                start:start + BATCH_SIZE
            ]

            tensor = torch.tensor(
                batch,
                dtype=torch.long,
                device=device
            )

            batch_scores = model.score(
                tensor[:, 0],
                tensor[:, 1],
                tensor[:, 2],
                tensor[:, 3]
            )

            scores.extend(
                batch_scores
                .cpu()
                .numpy()
                .tolist()
            )

    return scores


# ============================================================
# EXTRACT TEMPORAL FEATURES
# ============================================================

def extract_features(
    dataset_name,
    model,
    facts,
    label,
    time2id,
    id2time,
    device
):

    rows = []

    offsets = [
        -2,
        -1,
        0,
        1,
        2
    ]

    for start in range(
        0,
        len(facts),
        BATCH_SIZE
    ):

        batch = facts[
            start:start + BATCH_SIZE
        ]

        queries = []
        valid_flags = []

        # ====================================================
        # BUILD TEMPORAL QUERIES
        # ====================================================

        for h, r, o, t in batch:

            current_flags = []

            for delta in offsets:

                neighbor_t = get_neighbor_time_id(
                    dataset_name=dataset_name,
                    original_time_id=t,
                    delta=delta,
                    time2id=time2id,
                    id2time=id2time
                )

                # --------------------------------------------
                # Real temporal neighbor exists
                # --------------------------------------------

                if neighbor_t is not None:

                    queries.append(
                        (
                            h,
                            r,
                            o,
                            neighbor_t
                        )
                    )

                    current_flags.append(
                        True
                    )

                # --------------------------------------------
                # Neighbor absent from temporal vocabulary
                #
                # Use t only as temporary placeholder so that
                # batching works.
                #
                # Score will later be replaced by NaN.
                # --------------------------------------------

                else:

                    queries.append(
                        (
                            h,
                            r,
                            o,
                            t
                        )
                    )

                    current_flags.append(
                        False
                    )

            valid_flags.append(
                current_flags
            )

        # ====================================================
        # QUERY TeRo
        # ====================================================

        scores = score_queries(
            model=model,
            queries=queries,
            device=device
        )

        scores = torch.tensor(
            scores,
            dtype=torch.float32
        ).reshape(
            len(batch),
            5
        )

        # ====================================================
        # BUILD OUTPUT ROWS
        # ====================================================

        for i, (h, r, o, t) in enumerate(batch):

            values = (
                scores[i]
                .numpy()
                .tolist()
            )

            # Replace missing real temporal neighbors with NaN
            for j in range(5):

                if not valid_flags[i][j]:
                    values[j] = float("nan")

            rows.append(
                {
                    "h": h,
                    "r": r,
                    "o": o,
                    "t": t,

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
                        label
                }
            )

    return rows


# ============================================================
# BALANCE MEMBERS / NON-MEMBERS
# ============================================================

def balance_facts(
    member_facts,
    nonmember_facts,
    seed=42
):

    n = min(
        len(member_facts),
        len(nonmember_facts)
    )

    # --------------------------------------------------------
    # Random sampling rather than taking first n members.
    # Reproducible with fixed seed.
    # --------------------------------------------------------

    rng = random.Random(seed)

    member_facts = rng.sample(
        member_facts,
        n
    )

    nonmember_facts = rng.sample(
        nonmember_facts,
        n
    )

    return (
        member_facts,
        nonmember_facts
    )


# ============================================================
# BUILD ONE ATTACK DATASET
# ============================================================

def build_attack_dataset(
    dataset_name,
    model_type,
    member_filename,
    nonmember_filename,
    output_filename,
    device
):

    print("\n" + "=" * 75)

    print(
        f"{dataset_name} | {model_type.upper()}"
    )

    print("=" * 75)

    dataset_dir = (
        PROCESSED_ROOT
        /
        dataset_name
    )

    # --------------------------------------------------------
    # Load temporal mapping
    # --------------------------------------------------------

    time2id, id2time = load_time_mapping(
        dataset_name
    )

    print(
        f"Temporal vocabulary : {len(time2id)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        dataset_name=dataset_name,
        model_type=model_type,
        device=device
    )

    print(
        f"Loaded {model_type} model"
    )

    # --------------------------------------------------------
    # Load member/non-member candidates
    # --------------------------------------------------------

    member_facts = load_facts(
        dataset_dir
        /
        member_filename
    )

    nonmember_facts = load_facts(
        dataset_dir
        /
        nonmember_filename
    )

    print(
        f"Original members     : "
        f"{len(member_facts)}"
    )

    print(
        f"Original non-members : "
        f"{len(nonmember_facts)}"
    )

    # --------------------------------------------------------
    # Balance classes
    # --------------------------------------------------------

    (
        member_facts,
        nonmember_facts
    ) = balance_facts(
        member_facts,
        nonmember_facts,
        seed=SEED
    )

    print(
        f"Balanced             : "
        f"{len(member_facts)} + "
        f"{len(nonmember_facts)}"
    )

    # --------------------------------------------------------
    # Extract member features
    # --------------------------------------------------------

    print(
        "Scoring members..."
    )

    member_rows = extract_features(
        dataset_name=dataset_name,
        model=model,
        facts=member_facts,
        label=1,
        time2id=time2id,
        id2time=id2time,
        device=device
    )

    # --------------------------------------------------------
    # Extract non-member features
    # --------------------------------------------------------

    print(
        "Scoring non-members..."
    )

    nonmember_rows = extract_features(
        dataset_name=dataset_name,
        model=model,
        facts=nonmember_facts,
        label=0,
        time2id=time2id,
        id2time=id2time,
        device=device
    )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        member_rows
        +
        nonmember_rows
    )

    df = df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_dir = (
        OUTPUT_ROOT
        /
        dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        /
        output_filename
    )

    df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    temporal_cols = [
        "score_t_minus_2",
        "score_t_minus_1",
        "score_t",
        "score_t_plus_1",
        "score_t_plus_2"
    ]

    print(
        f"\nSamples               : {len(df)}"
    )

    print(
        f"Members               : "
        f"{(df['label'] == 1).sum()}"
    )

    print(
        f"Non-members           : "
        f"{(df['label'] == 0).sum()}"
    )

    print(
        f"score_t NaNs          : "
        f"{df['score_t'].isna().sum()}"
    )

    print(
        f"Rows with temporal NaN: "
        f"{df[temporal_cols].isna().any(axis=1).sum()}"
    )

    print("\nNaN by temporal feature:")

    for col in temporal_cols:

        print(
            f"  {col:18s}: "
            f"{df[col].isna().sum()}"
        )

    print(
        f"\nSaved: {output_path}"
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

    print("=" * 75)
    print("CORRECTED TEMPORAL ATTACK DATASET GENERATION")
    print("=" * 75)

    print(
        f"Device: {device}"
    )

    successful = []
    failed = []

    for dataset_name in DATASETS:

        try:

            # =================================================
            # SHADOW ATTACK DATASET
            # =================================================

            build_attack_dataset(
                dataset_name=dataset_name,
                model_type="shadow",
                member_filename="shadow_train.txt",
                nonmember_filename="shadow_test.txt",
                output_filename=
                    "shadow_attack_dataset.csv",
                device=device
            )

            # =================================================
            # VICTIM ATTACK DATASET
            # =================================================

            build_attack_dataset(
                dataset_name=dataset_name,
                model_type="victim",
                member_filename="victim_train.txt",
                nonmember_filename="victim_test.txt",
                output_filename=
                    "victim_attack_dataset.csv",
                device=device
            )

            successful.append(
                dataset_name
            )

        except Exception as e:

            print(
                f"\nERROR for {dataset_name}:"
            )

            print(
                repr(e)
            )

            failed.append(
                dataset_name
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("FINISHED")
    print("=" * 75)

    print("\nSuccessful:")

    for dataset_name in successful:
        print(
            f"  ✓ {dataset_name}"
        )

    print("\nFailed:")

    if failed:

        for dataset_name in failed:
            print(
                f"  ✗ {dataset_name}"
            )

    else:

        print(
            "  None"
        )