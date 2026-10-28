from pathlib import Path
import json
import csv


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("/home/harrouch/Paper2/Paper2")

PROCESSED = ROOT / "processed"
OUTPUT = ROOT / "dkg"

DATASETS = [
    "ICEWS05-15",
    "ICEWS14",
    "YAGO11k",
    "WIKIDATA12k",
]

# Candidate chronological splits:
# 70% history / 30% update
# 80% history / 20% update
# 90% history / 10% update
CANDIDATE_RATIOS = [0.70, 0.80, 0.90]

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# READING DATA
# ============================================================

def read_quadruples(path):
    """
    Read a temporal KG file.

    Expected format:
        h r o t

    The fourth column t can be:
        - a real timestamp, e.g. 2014-01-01
        - OR a temporal ID, e.g. 0, 1, 2, ...

    Returns:
        list of tuples:
        (h, r, o, t)
    """

    rows = []

    with open(path, "r", encoding="utf-8") as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 4:
                raise ValueError(
                    f"{path}:{line_no}: "
                    f"expected at least 4 columns, "
                    f"got {len(parts)}"
                )

            h, r, o, t = parts[:4]

            rows.append(
                (
                    str(h),
                    str(r),
                    str(o),
                    str(t),
                )
            )

    return rows


def load_victim_data(dataset):
    """
    Reconstruct victim-side temporal data from:

        victim_train.txt
        victim_test.txt

    Shadow data are NOT used here.

    Exact duplicate quadruples are removed.
    """

    dataset_dir = PROCESSED / dataset

    train_path = dataset_dir / "victim_train.txt"
    test_path = dataset_dir / "victim_test.txt"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing file: {train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing file: {test_path}"
        )

    train = read_quadruples(train_path)
    test = read_quadruples(test_path)

    # Remove exact duplicate quadruples while preserving order.
    all_rows = list(
        dict.fromkeys(train + test)
    )

    return all_rows


# ============================================================
# TEMPORAL VOCABULARY
# ============================================================

def load_time_information(dataset):
    """
    Load time2id.json.

    Example:

        {
            "2005-01-01": 0,
            "2005-01-02": 1
        }

    Our processed files may contain either:

        2005-01-01

    OR:

        0

    in their fourth column.

    Therefore we build a mapping supporting BOTH representations.

    Returns:
        time_order:
            maps timestamp representation -> chronological ID

        id_to_original:
            maps chronological ID -> original timestamp
    """

    path = PROCESSED / dataset / "time2id.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw_time2id = json.load(f)

    time_order = {}
    id_to_original = {}

    for original_timestamp, time_id in raw_time2id.items():

        time_id = int(time_id)

        original_timestamp = str(original_timestamp)

        # Real timestamp representation
        time_order[original_timestamp] = time_id

        # Encoded-ID representation
        time_order[str(time_id)] = time_id

        # Reverse mapping
        id_to_original[time_id] = original_timestamp

    return time_order, id_to_original


def get_original_timestamp(t, time_order, id_to_original):
    """
    Convert whatever representation appears in the processed
    file back to the original date/year when possible.
    """

    t = str(t)

    if t not in time_order:
        return t

    time_id = time_order[t]

    return id_to_original.get(
        time_id,
        t
    )


# ============================================================
# CUTOFF EVALUATION
# ============================================================

def evaluate_cutoff(
    rows,
    time_order,
    id_to_original,
    ratio
):
    """
    Build a chronological cumulative DKG split.

    --------------------------------------------------------
    G1
    --------------------------------------------------------

    Contains observations belonging to the first `ratio`
    fraction of UNIQUE timestamps.

    Example:
        ratio = 0.80

        first 80% timestamps -> G1
        last 20% timestamps  -> update

    --------------------------------------------------------
    delta_G
    --------------------------------------------------------

    Contains observations after the cutoff.

    --------------------------------------------------------
    G2
    --------------------------------------------------------

        G2 = G1 union delta_G

    It is therefore cumulative.

    --------------------------------------------------------
    ADDED
    --------------------------------------------------------

    A base triple (h,r,o) is Added if:

        (h,r,o) not in G1

    AND

        (h,r,o) appears after cutoff.

    For the output added.txt, we retain the FIRST
    post-cutoff occurrence of that base triple.

    --------------------------------------------------------
    UNCHANGED
    --------------------------------------------------------

    A base triple (h,r,o) is Unchanged if it already
    exists in G1.

    Since G2 is cumulative, it remains present in G2.

    For unchanged.txt, we retain its LAST occurrence
    inside G1.
    """

    # --------------------------------------------------------
    # Get unique timestamps actually present in this dataset
    # --------------------------------------------------------

    present_times = {
        str(t)
        for _, _, _, t in rows
    }

    # Validate all timestamps
    missing = [
        t
        for t in present_times
        if t not in time_order
    ]

    if missing:

        missing = sorted(missing)

        raise ValueError(
            f"{len(missing)} timestamps cannot be mapped. "
            f"Examples: {missing[:10]}"
        )

    # Chronological ordering
    present_times = sorted(
        present_times,
        key=lambda x: time_order[x]
    )

    n_times = len(present_times)

    if n_times < 2:
        raise ValueError(
            "Dataset contains fewer than 2 timestamps."
        )

    # --------------------------------------------------------
    # Find split index
    # --------------------------------------------------------

    split_idx = int(
        n_times * ratio
    )

    # Ensure at least one timestamp on each side
    split_idx = max(
        1,
        min(
            split_idx,
            n_times - 1
        )
    )

    g1_times = set(
        present_times[:split_idx]
    )

    update_times = set(
        present_times[split_idx:]
    )

    cutoff_raw = present_times[
        split_idx - 1
    ]

    first_update_raw = present_times[
        split_idx
    ]

    cutoff_id = time_order[
        cutoff_raw
    ]

    first_update_id = time_order[
        first_update_raw
    ]

    cutoff_original = id_to_original.get(
        cutoff_id,
        cutoff_raw
    )

    first_update_original = id_to_original.get(
        first_update_id,
        first_update_raw
    )

    # --------------------------------------------------------
    # Split rows
    # --------------------------------------------------------

    g1_rows = [
        q
        for q in rows
        if q[3] in g1_times
    ]

    delta_rows = [
        q
        for q in rows
        if q[3] in update_times
    ]

    # Cumulative G2
    g2_rows = (
        g1_rows
        +
        delta_rows
    )

    # --------------------------------------------------------
    # Base triples
    # --------------------------------------------------------

    g1_triples = {
        (h, r, o)
        for h, r, o, t
        in g1_rows
    }

    delta_triples = {
        (h, r, o)
        for h, r, o, t
        in delta_rows
    }

    g2_triples = (
        g1_triples
        |
        delta_triples
    )

    # --------------------------------------------------------
    # ADDED
    # --------------------------------------------------------

    added_triples = (
        delta_triples
        -
        g1_triples
    )

    # --------------------------------------------------------
    # UNCHANGED
    # --------------------------------------------------------

    # Because G2 is cumulative:
    #
    # every triple in G1 remains in G2.
    #
    # Still write the definition explicitly.
    unchanged_triples = (
        g1_triples
        &
        g2_triples
    )

    # --------------------------------------------------------
    # First occurrence of Added facts after cutoff
    # --------------------------------------------------------

    first_added_occurrence = {}

    sorted_delta = sorted(
        delta_rows,
        key=lambda q: time_order[q[3]]
    )

    for h, r, o, t in sorted_delta:

        triple = (
            h,
            r,
            o
        )

        if (
            triple in added_triples
            and
            triple not in first_added_occurrence
        ):

            first_added_occurrence[
                triple
            ] = (
                h,
                r,
                o,
                t
            )

    # --------------------------------------------------------
    # Last occurrence of Unchanged facts before cutoff
    # --------------------------------------------------------

    last_unchanged_occurrence = {}

    sorted_g1 = sorted(
        g1_rows,
        key=lambda q: time_order[q[3]]
    )

    for h, r, o, t in sorted_g1:

        triple = (
            h,
            r,
            o
        )

        if triple in unchanged_triples:

            last_unchanged_occurrence[
                triple
            ] = (
                h,
                r,
                o,
                t
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    n_added = len(
        first_added_occurrence
    )

    n_unchanged = len(
        last_unchanged_occurrence
    )

    balanced_capacity = min(
        n_added,
        n_unchanged
    )

    actual_update_ratio = (
        len(update_times)
        /
        n_times
    )

    target_update_ratio = 0.20

    temporal_penalty = abs(
        actual_update_ratio
        -
        target_update_ratio
    )

    return {

        # Requested split
        "ratio": ratio,

        # Temporal boundary
        "cutoff_raw": cutoff_raw,
        "cutoff_id": cutoff_id,
        "cutoff_original": cutoff_original,

        "first_update_raw": first_update_raw,
        "first_update_id": first_update_id,
        "first_update_original": first_update_original,

        # Timestamp counts
        "n_timestamps_total": n_times,
        "n_timestamps_g1": len(
            g1_times
        ),
        "n_timestamps_update": len(
            update_times
        ),

        # Quadruple counts
        "n_rows_total": len(
            rows
        ),
        "n_rows_g1": len(
            g1_rows
        ),
        "n_rows_delta": len(
            delta_rows
        ),
        "n_rows_g2": len(
            g2_rows
        ),

        # Triple counts
        "n_unique_triples_g1": len(
            g1_triples
        ),
        "n_unique_triples_delta": len(
            delta_triples
        ),
        "n_unique_triples_g2": len(
            g2_triples
        ),

        # Attack populations
        "n_added": n_added,
        "n_unchanged": n_unchanged,

        "balanced_capacity": (
            balanced_capacity
        ),

        "actual_update_ratio": (
            actual_update_ratio
        ),

        "temporal_penalty": (
            temporal_penalty
        ),

        # Internal objects
        "_g1_rows": g1_rows,
        "_delta_rows": delta_rows,
        "_g2_rows": g2_rows,

        "_added": list(
            first_added_occurrence.values()
        ),

        "_unchanged": list(
            last_unchanged_occurrence.values()
        ),
    }


# ============================================================
# CHOOSE BEST CUTOFF
# ============================================================

def choose_best(results):
    """
    Choose the best candidate split.

    Main priority:
        maximize balanced Added/Unchanged capacity.

    Secondary priority:
        prefer an update window around 20%.

    Selection score:

        90% usable attack population
        10% temporal-window preference

    IMPORTANT:
    This is a heuristic selection criterion, not a claim
    of mathematically optimal splitting.
    """

    max_capacity = max(
        r["balanced_capacity"]
        for r in results
    )

    if max_capacity == 0:

        raise RuntimeError(
            "No Added examples were found "
            "for any candidate cutoff."
        )

    for r in results:

        capacity_norm = (
            r["balanced_capacity"]
            /
            max_capacity
        )

        temporal_component = max(
            0.0,
            1.0 - r["temporal_penalty"]
        )

        r["selection_score"] = (
            0.90 * capacity_norm
            +
            0.10 * temporal_component
        )

    return max(
        results,
        key=lambda x: x["selection_score"]
    )


# ============================================================
# OUTPUT
# ============================================================

def write_rows(path, rows):
    """
    Save quadruples as:
        h TAB r TAB o TAB t
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for h, r, o, t in rows:

            f.write(
                f"{h}\t{r}\t{o}\t{t}\n"
            )


def clean_result_for_json(result):
    """
    Remove internal lists beginning with "_"
    before saving JSON statistics.
    """

    return {
        k: v
        for k, v in result.items()
        if not k.startswith("_")
    }


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(dataset):

    print()
    print("=" * 90)
    print(f"DATASET: {dataset}")
    print("=" * 90)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    rows = load_victim_data(
        dataset
    )

    time_order, id_to_original = (
        load_time_information(
            dataset
        )
    )

    # --------------------------------------------------------
    # Global validation
    # --------------------------------------------------------

    observed_times = {
        t
        for _, _, _, t
        in rows
    }

    missing = sorted(
        t
        for t in observed_times
        if t not in time_order
    )

    if missing:

        raise ValueError(
            f"{dataset}: "
            f"{len(missing)} timestamps missing "
            f"from temporal mapping. "
            f"Examples: {missing[:10]}"
        )

    print(
        f"Loaded quadruples       : "
        f"{len(rows):,}"
    )

    print(
        f"Observed timestamps     : "
        f"{len(observed_times):,}"
    )

    print(
        f"Temporal vocabulary     : "
        f"{len(id_to_original):,}"
    )

    # --------------------------------------------------------
    # Evaluate candidate cutoffs
    # --------------------------------------------------------

    results = []

    for ratio in CANDIDATE_RATIOS:

        result = evaluate_cutoff(
            rows,
            time_order,
            id_to_original,
            ratio
        )

        results.append(
            result
        )

        history_pct = int(
            ratio * 100
        )

        update_pct = int(
            round(
                (1 - ratio) * 100
            )
        )

        print()
        print(
            f"Candidate "
            f"{history_pct}/{update_pct}"
        )

        print(
            f"  cutoff ID              : "
            f"{result['cutoff_id']}"
        )

        print(
            f"  cutoff date/year       : "
            f"{result['cutoff_original']}"
        )

        print(
            f"  first update ID        : "
            f"{result['first_update_id']}"
        )

        print(
            f"  first update date/year : "
            f"{result['first_update_original']}"
        )

        print(
            f"  timestamps G1          : "
            f"{result['n_timestamps_g1']:,}"
        )

        print(
            f"  timestamps update      : "
            f"{result['n_timestamps_update']:,}"
        )

        print(
            f"  G1 rows                : "
            f"{result['n_rows_g1']:,}"
        )

        print(
            f"  update rows            : "
            f"{result['n_rows_delta']:,}"
        )

        print(
            f"  G1 unique triples      : "
            f"{result['n_unique_triples_g1']:,}"
        )

        print(
            f"  Added                  : "
            f"{result['n_added']:,}"
        )

        print(
            f"  Unchanged              : "
            f"{result['n_unchanged']:,}"
        )

        print(
            f"  balanced capacity      : "
            f"{result['balanced_capacity']:,}"
        )

    # --------------------------------------------------------
    # Choose best
    # --------------------------------------------------------

    best = choose_best(
        results
    )

    print()
    print("-" * 90)
    print("SELECTED SPLIT")
    print("-" * 90)

    history_pct = int(
        best["ratio"] * 100
    )

    update_pct = int(
        round(
            (1 - best["ratio"]) * 100
        )
    )

    print(
        f"Temporal split          : "
        f"{history_pct}/{update_pct}"
    )

    print(
        f"Cutoff ID               : "
        f"{best['cutoff_id']}"
    )

    print(
        f"Cutoff date/year        : "
        f"{best['cutoff_original']}"
    )

    print(
        f"First update ID         : "
        f"{best['first_update_id']}"
    )

    print(
        f"First update date/year  : "
        f"{best['first_update_original']}"
    )

    print(
        f"Added                   : "
        f"{best['n_added']:,}"
    )

    print(
        f"Unchanged               : "
        f"{best['n_unchanged']:,}"
    )

    print(
        f"Balanced capacity       : "
        f"{best['balanced_capacity']:,}"
    )

    print(
        f"Selection score         : "
        f"{best['selection_score']:.6f}"
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    dataset_out = (
        OUTPUT
        /
        dataset
    )

    dataset_out.mkdir(
        parents=True,
        exist_ok=True
    )

    write_rows(
        dataset_out / "G1.txt",
        best["_g1_rows"]
    )

    write_rows(
        dataset_out / "delta_G.txt",
        best["_delta_rows"]
    )

    write_rows(
        dataset_out / "G2.txt",
        best["_g2_rows"]
    )

    write_rows(
        dataset_out / "added.txt",
        best["_added"]
    )

    write_rows(
        dataset_out / "unchanged.txt",
        best["_unchanged"]
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    candidate_stats = [
        clean_result_for_json(r)
        for r in results
    ]

    info = {

        "dataset": dataset,

        "source": {
            "directory": str(
                PROCESSED / dataset
            ),
            "files": [
                "victim_train.txt",
                "victim_test.txt",
                "time2id.json",
            ],
        },

        "dkg_definition": {

            "G1": (
                "All timestamped observations "
                "up to the selected chronological cutoff."
            ),

            "delta_G": (
                "All timestamped observations "
                "after the selected cutoff."
            ),

            "G2": (
                "Cumulative graph: G1 union delta_G."
            ),

            "Added": (
                "A base triple (h,r,o) absent from G1 "
                "whose first occurrence is after the cutoff."
            ),

            "Unchanged": (
                "A base triple (h,r,o) already present in G1 "
                "and retained in cumulative G2."
            ),
        },

        "candidate_ratios": (
            CANDIDATE_RATIOS
        ),

        "selected_ratio": (
            best["ratio"]
        ),

        "selected_cutoff": {

            "id": (
                best["cutoff_id"]
            ),

            "original_timestamp": (
                best["cutoff_original"]
            ),
        },

        "first_update_timestamp": {

            "id": (
                best["first_update_id"]
            ),

            "original_timestamp": (
                best["first_update_original"]
            ),
        },

        "selected_statistics": (
            clean_result_for_json(
                best
            )
        ),

        "candidate_splits": (
            candidate_stats
        ),
    }

    info_path = (
        dataset_out
        /
        "split_info.json"
    )

    with open(
        info_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            info,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Return global summary
    # --------------------------------------------------------

    return {

        "dataset": dataset,

        "selected_ratio": (
            best["ratio"]
        ),

        "cutoff_id": (
            best["cutoff_id"]
        ),

        "cutoff_timestamp": (
            best["cutoff_original"]
        ),

        "first_update_id": (
            best["first_update_id"]
        ),

        "first_update_timestamp": (
            best["first_update_original"]
        ),

        "timestamps_total": (
            best["n_timestamps_total"]
        ),

        "G1_rows": (
            best["n_rows_g1"]
        ),

        "delta_rows": (
            best["n_rows_delta"]
        ),

        "G2_rows": (
            best["n_rows_g2"]
        ),

        "G1_unique_triples": (
            best["n_unique_triples_g1"]
        ),

        "Added": (
            best["n_added"]
        ),

        "Unchanged": (
            best["n_unchanged"]
        ),

        "balanced_capacity": (
            best["balanced_capacity"]
        ),

        "selection_score": (
            best["selection_score"]
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    summaries = []

    for dataset in DATASETS:

        try:

            summary = process_dataset(
                dataset
            )

            summaries.append(
                summary
            )

        except Exception as e:

            print()
            print("!" * 90)

            print(
                f"ERROR while processing "
                f"{dataset}"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            print("!" * 90)

            raise

    # --------------------------------------------------------
    # Save global CSV summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT
        /
        "dkg_split_summary.csv"
    )

    if summaries:

        with open(
            summary_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    summaries[0].keys()
                )
            )

            writer.writeheader()

            writer.writerows(
                summaries
            )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("FINAL SUMMARY")
    print("=" * 110)

    for s in summaries:

        history_pct = int(
            s["selected_ratio"]
            * 100
        )

        update_pct = int(
            round(
                (
                    1
                    -
                    s["selected_ratio"]
                )
                * 100
            )
        )

        print(
            f"{s['dataset']:15s} | "
            f"split={history_pct}/{update_pct} | "
            f"cutoff={s['cutoff_timestamp']} | "
            f"Added={s['Added']:,} | "
            f"Unchanged={s['Unchanged']:,} | "
            f"capacity={s['balanced_capacity']:,}"
        )

    print()
    print(
        f"Results saved in:"
    )

    print(
        OUTPUT
    )

    print()

    print(
        f"Global summary:"
    )

    print(
        summary_path
    )

    print()
    print(
        "IMPORTANT: inspect these statistics before "
        "training M1/M2."
    )


if __name__ == "__main__":
    main()