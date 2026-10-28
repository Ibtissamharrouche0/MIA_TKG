from pathlib import Path
import csv
import json
import shutil
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("/home/harrouch/Paper2/Paper2")
DATA_ROOT = ROOT / "data"
OUTPUT_ROOT = ROOT / "dkg"

DATASETS = {
    "YAGO11k": DATA_ROOT / "YAGO11k" / "full_graph.txt",
    "WIKIDATA12k": DATA_ROOT / "WIKIDATA12k" / "full_graph.txt",
}

# Candidate positions in the observed temporal history.
#
# We test several t1 -> t2 snapshot pairs.
#
# Example:
#   (0.70, 0.80)
# means:
#   t1 = timestamp around 70% of the timeline
#   t2 = timestamp around 80% of the timeline
#
# This is NOT a random split.
SNAPSHOT_PAIRS = [
    (0.60, 0.70),
    (0.60, 0.80),
    (0.70, 0.80),
    (0.70, 0.90),
    (0.80, 0.90),
]

# Unknown end date is interpreted as right-censored:
# no end was observed in the dataset.
OPEN_END = None


# ============================================================
# DATE / YEAR PARSING
# ============================================================

def parse_year(value):
    """
    Extract a year from temporal strings such as:

        1952-##-##
        1925-04-23
        ####-##-##

    Returns:
        int year
        OR None if the year is unknown.

    We intentionally work at YEAR granularity because YAGO11k
    and WIKIDATA12k contain partially specified dates.
    """

    value = value.strip()

    if not value:
        return None

    year_part = value.split("-")[0]

    if "#" in year_part:
        return None

    try:
        return int(year_part)

    except ValueError:
        return None


# ============================================================
# READ ORIGINAL INTERVAL DATA
# ============================================================

def read_interval_graph(path):
    """
    Expected format:

        h  r  o  start_date  end_date

    Example:

        6464 5 6465 1952-##-## 1964-##-##
        10289 9 10290 1787-##-## ####-##-##

    Returns one record per original interval fact.
    """

    records = []

    stats = Counter()

    with open(path, "r", encoding="utf-8") as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                raise ValueError(
                    f"{path}:{line_no}: expected at least "
                    f"5 columns, got {len(parts)}"
                )

            h, r, o, start_raw, end_raw = parts[:5]

            start_year = parse_year(start_raw)
            end_year = parse_year(end_raw)

            stats["total_rows"] += 1

            if start_year is None:
                stats["unknown_start"] += 1

            if end_year is None:
                stats["unknown_end"] += 1

            # ------------------------------------------------
            # We need a known start to determine whether a fact
            # is newly Added between t1 and t2.
            #
            # Therefore facts with unknown start are excluded
            # from this first Added/Unchanged experiment.
            # ------------------------------------------------

            if start_year is None:
                stats["excluded_unknown_start"] += 1
                continue

            # Basic sanity check.
            if (
                end_year is not None
                and end_year < start_year
            ):
                stats["excluded_invalid_interval"] += 1
                continue

            records.append({
                "h": str(h),
                "r": str(r),
                "o": str(o),

                "start_raw": start_raw,
                "end_raw": end_raw,

                "start_year": start_year,
                "end_year": end_year,
            })

            stats["usable_rows"] += 1

    return records, stats


# ============================================================
# INTERVAL SEMANTICS
# ============================================================

def is_active(record, year):
    """
    A fact is active at year t if:

        start <= t <= end

    For an unknown end:

        start <= t

    because the end is right-censored / not observed.
    """

    start = record["start_year"]
    end = record["end_year"]

    if start > year:
        return False

    if end is None:
        return True

    return year <= end


def triple(record):
    return (
        record["h"],
        record["r"],
        record["o"],
    )


# ============================================================
# BUILD SNAPSHOT
# ============================================================

def build_snapshot(records, year):
    """
    Return base triples active at a given year.

    Multiple interval records corresponding to the same
    (h,r,o) are collapsed to one base triple.
    """

    active = {}

    for record in records:

        if is_active(record, year):

            q = triple(record)

            # Keep one representative original record.
            if q not in active:
                active[q] = record

    return active


# ============================================================
# SELECT REPRESENTATIVE RECORDS
# ============================================================

def choose_added_record(records_by_triple, q, t1, t2):
    """
    For an Added triple, choose a record demonstrating that
    the triple becomes active after t1 and is active at t2.

    Preference:
        latest start <= t2 among starts > t1.

    If the base triple becomes active at t2 through a more
    complicated interval pattern, fall back to an interval
    active at t2.
    """

    candidates = records_by_triple[q]

    strict_added = [
        r for r in candidates
        if (
            r["start_year"] > t1
            and
            r["start_year"] <= t2
            and
            is_active(r, t2)
        )
    ]

    if strict_added:

        return min(
            strict_added,
            key=lambda r: r["start_year"]
        )

    active_t2 = [
        r for r in candidates
        if is_active(r, t2)
    ]

    if not active_t2:
        raise RuntimeError(
            f"Cannot find active t2 record for Added triple {q}"
        )

    return min(
        active_t2,
        key=lambda r: r["start_year"]
    )


def choose_unchanged_record(records_by_triple, q, t1, t2):
    """
    Choose an interval record supporting that the triple
    is active at BOTH t1 and t2.

    Ideally one single interval spans both snapshots.
    """

    candidates = records_by_triple[q]

    spanning = [
        r for r in candidates
        if (
            is_active(r, t1)
            and
            is_active(r, t2)
        )
    ]

    if spanning:

        return max(
            spanning,
            key=lambda r: r["start_year"]
        )

    # Base-triple-level fallback:
    # the triple is present in both snapshots even if separate
    # interval records represent its temporal history.
    active_t1 = [
        r for r in candidates
        if is_active(r, t1)
    ]

    if not active_t1:
        raise RuntimeError(
            f"Cannot find t1 record for Unchanged triple {q}"
        )

    return max(
        active_t1,
        key=lambda r: r["start_year"]
    )


# ============================================================
# INDEX RECORDS BY BASE TRIPLE
# ============================================================

def index_by_triple(records):

    result = {}

    for record in records:

        q = triple(record)

        result.setdefault(
            q,
            []
        ).append(record)

    return result


# ============================================================
# CANDIDATE SNAPSHOT YEARS
# ============================================================

def get_observed_years(records):
    """
    Build temporal domain from all known interval boundaries.

    Both known starts and known ends are included.
    """

    years = set()

    for record in records:

        years.add(
            record["start_year"]
        )

        if record["end_year"] is not None:
            years.add(
                record["end_year"]
            )

    return sorted(years)


def percentile_year(years, ratio):
    """
    Select an observed year corresponding approximately
    to a position in the temporal history.
    """

    if not years:
        raise ValueError(
            "No temporal years available."
        )

    idx = int(
        ratio * (len(years) - 1)
    )

    idx = max(
        0,
        min(
            idx,
            len(years) - 1
        )
    )

    return years[idx]


# ============================================================
# EVALUATE ONE SNAPSHOT PAIR
# ============================================================

def evaluate_pair(
    records,
    records_by_triple,
    years,
    ratio_t1,
    ratio_t2
):

    t1 = percentile_year(
        years,
        ratio_t1
    )

    t2 = percentile_year(
        years,
        ratio_t2
    )

    if t2 <= t1:
        return None

    G1 = build_snapshot(
        records,
        t1
    )

    G2 = build_snapshot(
        records,
        t2
    )

    G1_triples = set(
        G1.keys()
    )

    G2_triples = set(
        G2.keys()
    )

    # ========================================================
    # CORE DKG DEFINITIONS
    # ========================================================

    added_triples = (
        G2_triples
        -
        G1_triples
    )

    unchanged_triples = (
        G1_triples
        &
        G2_triples
    )

    removed_triples = (
        G1_triples
        -
        G2_triples
    )

    # Removed is NOT part of the attack.
    # We report it only to understand snapshot evolution.

    added_records = []

    for q in sorted(added_triples):

        record = choose_added_record(
            records_by_triple,
            q,
            t1,
            t2
        )

        added_records.append(
            record
        )

    unchanged_records = []

    for q in sorted(unchanged_triples):

        record = choose_unchanged_record(
            records_by_triple,
            q,
            t1,
            t2
        )

        unchanged_records.append(
            record
        )

    balanced_capacity = min(
        len(added_records),
        len(unchanged_records)
    )

    temporal_gap = (
        t2 - t1
    )

    return {
        "ratio_t1": ratio_t1,
        "ratio_t2": ratio_t2,

        "t1": t1,
        "t2": t2,

        "temporal_gap_years": temporal_gap,

        "G1_size": len(
            G1_triples
        ),

        "G2_size": len(
            G2_triples
        ),

        "Added": len(
            added_triples
        ),

        "Unchanged": len(
            unchanged_triples
        ),

        "Removed_report_only": len(
            removed_triples
        ),

        "balanced_capacity": (
            balanced_capacity
        ),

        "_G1": G1,
        "_G2": G2,

        "_added_records": (
            added_records
        ),

        "_unchanged_records": (
            unchanged_records
        ),

        "_removed": (
            removed_triples
        ),
    }


# ============================================================
# CHOOSE BEST SNAPSHOT PAIR
# ============================================================

def choose_best(results):
    """
    Main criterion:
        maximize min(#Added, #Unchanged)

    This gives us the largest possible balanced attack
    population.

    We do NOT use attack performance to choose the split.
    """

    valid = [
        r for r in results
        if (
            r is not None
            and
            r["Added"] > 0
            and
            r["Unchanged"] > 0
        )
    ]

    if not valid:
        raise RuntimeError(
            "No candidate snapshot pair produced both "
            "Added and Unchanged facts."
        )

    max_capacity = max(
        r["balanced_capacity"]
        for r in valid
    )

    # Among pairs with maximum capacity, prefer the smaller
    # temporal gap. This avoids choosing an unnecessarily
    # huge evolution window.
    best_candidates = [
        r for r in valid
        if r["balanced_capacity"] == max_capacity
    ]

    best = min(
        best_candidates,
        key=lambda r: r["temporal_gap_years"]
    )

    return best


# ============================================================
# WRITING OUTPUTS
# ============================================================

def write_snapshot(path, snapshot, year):
    """
    Write snapshot as:

        h r o snapshot_year

    This is a state representation for the DKG experiment.
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for h, r, o in sorted(snapshot.keys()):

            f.write(
                f"{h}\t{r}\t{o}\t{year}\n"
            )


def write_interval_records(path, records):
    """
    Preserve original interval representation:

        h r o original_start original_end
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for r in records:

            f.write(
                f"{r['h']}\t"
                f"{r['r']}\t"
                f"{r['o']}\t"
                f"{r['start_raw']}\t"
                f"{r['end_raw']}\n"
            )


def write_attack_candidates(
    path,
    added_records,
    unchanged_records,
    t1,
    t2
):
    """
    Write all candidates with labels.

    label:
        1 = Added
        0 = Unchanged

    NOTE:
    This is NOT yet the final balanced/matched attack dataset.
    It is the candidate population.
    """

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "h",
            "r",
            "o",
            "start",
            "end",
            "t1",
            "t2",
            "class",
            "label",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for r in added_records:

            writer.writerow({
                "h": r["h"],
                "r": r["r"],
                "o": r["o"],
                "start": r["start_raw"],
                "end": r["end_raw"],
                "t1": t1,
                "t2": t2,
                "class": "Added",
                "label": 1,
            })

        for r in unchanged_records:

            writer.writerow({
                "h": r["h"],
                "r": r["r"],
                "o": r["o"],
                "start": r["start_raw"],
                "end": r["end_raw"],
                "t1": t1,
                "t2": t2,
                "class": "Unchanged",
                "label": 0,
            })


# ============================================================
# CLEAN JSON RESULT
# ============================================================

def clean_result(result):

    return {
        k: v
        for k, v in result.items()
        if not k.startswith("_")
    }


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(dataset, input_path):

    print()
    print("=" * 100)
    print(f"DATASET: {dataset}")
    print("=" * 100)

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file does not exist:\n{input_path}"
        )

    # ========================================================
    # LOAD ORIGINAL INTERVAL DATA
    # ========================================================

    records, raw_stats = (
        read_interval_graph(
            input_path
        )
    )

    print(
        f"Original rows          : "
        f"{raw_stats['total_rows']:,}"
    )

    print(
        f"Usable interval rows   : "
        f"{raw_stats['usable_rows']:,}"
    )

    print(
        f"Unknown start excluded : "
        f"{raw_stats['excluded_unknown_start']:,}"
    )

    print(
        f"Unknown/open end       : "
        f"{raw_stats['unknown_end']:,}"
    )

    print(
        f"Invalid intervals      : "
        f"{raw_stats['excluded_invalid_interval']:,}"
    )

    # ========================================================
    # INDEX
    # ========================================================

    records_by_triple = (
        index_by_triple(
            records
        )
    )

    years = get_observed_years(
        records
    )

    print(
        f"Unique base triples    : "
        f"{len(records_by_triple):,}"
    )

    print(
        f"Observed years         : "
        f"{len(years):,}"
    )

    print(
        f"Temporal range         : "
        f"{years[0]} -> {years[-1]}"
    )

    # ========================================================
    # TEST SNAPSHOT PAIRS
    # ========================================================

    results = []

    print()
    print("CANDIDATE SNAPSHOT PAIRS")
    print("-" * 100)

    for ratio_t1, ratio_t2 in SNAPSHOT_PAIRS:

        result = evaluate_pair(
            records,
            records_by_triple,
            years,
            ratio_t1,
            ratio_t2
        )

        if result is None:
            continue

        results.append(
            result
        )

        print(
            f"t1={result['t1']:5d} "
            f"({ratio_t1:.0%}) -> "
            f"t2={result['t2']:5d} "
            f"({ratio_t2:.0%}) | "
            f"G1={result['G1_size']:,} | "
            f"G2={result['G2_size']:,} | "
            f"Added={result['Added']:,} | "
            f"Unchanged={result['Unchanged']:,} | "
            f"Removed={result['Removed_report_only']:,} | "
            f"Capacity={result['balanced_capacity']:,}"
        )

    # ========================================================
    # CHOOSE BEST
    # ========================================================

    best = choose_best(
        results
    )

    print()
    print("-" * 100)
    print("SELECTED SNAPSHOT PAIR")
    print("-" * 100)

    print(
        f"t1                    : {best['t1']}"
    )

    print(
        f"t2                    : {best['t2']}"
    )

    print(
        f"Temporal gap          : "
        f"{best['temporal_gap_years']} years"
    )

    print(
        f"G1                    : "
        f"{best['G1_size']:,}"
    )

    print(
        f"G2                    : "
        f"{best['G2_size']:,}"
    )

    print(
        f"Added                 : "
        f"{best['Added']:,}"
    )

    print(
        f"Unchanged             : "
        f"{best['Unchanged']:,}"
    )

    print(
        f"Removed (report only) : "
        f"{best['Removed_report_only']:,}"
    )

    print(
        f"Balanced capacity     : "
        f"{best['balanced_capacity']:,}"
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = (
        OUTPUT_ROOT
        /
        dataset
    )

    # Delete ONLY this dataset's previous DKG output.
    #
    # ICEWS folders are never touched because this script
    # processes only YAGO11k and WIKIDATA12k.
    if output_dir.exists():

        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE SNAPSHOTS
    # ========================================================

    write_snapshot(
        output_dir / "G1.txt",
        best["_G1"],
        best["t1"]
    )

    write_snapshot(
        output_dir / "G2.txt",
        best["_G2"],
        best["t2"]
    )

    # ========================================================
    # SAVE ADDED / UNCHANGED
    # ========================================================

    write_interval_records(
        output_dir / "added.txt",
        best["_added_records"]
    )

    write_interval_records(
        output_dir / "unchanged.txt",
        best["_unchanged_records"]
    )

    # ========================================================
    # SAVE COMPLETE CANDIDATE POPULATION
    # ========================================================

    write_attack_candidates(
        output_dir / "attack_candidates.csv",
        best["_added_records"],
        best["_unchanged_records"],
        best["t1"],
        best["t2"]
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {

        "dataset": dataset,

        "input_file": str(
            input_path
        ),

        "temporal_semantics": (
            "Original interval-based facts. "
            "Snapshots contain base triples active "
            "at the selected year."
        ),

        "unknown_date_handling": {

            "unknown_start": (
                "Excluded because the start of validity "
                "cannot be determined."
            ),

            "unknown_end": (
                "Treated as right-censored: no termination "
                "is observed within the dataset."
            ),

            "partial_date": (
                "Year-level granularity is used. "
                "Unknown month/day are not imputed."
            ),
        },

        "attack_definition": {

            "G1": (
                "Base triples active at t1."
            ),

            "G2": (
                "Base triples active at t2."
            ),

            "Added": (
                "G2 - G1"
            ),

            "Unchanged": (
                "G1 intersection G2"
            ),

            "Removed": (
                "G1 - G2; reported for descriptive purposes "
                "only and not used in the current attack."
            ),
        },

        "raw_statistics": dict(
            raw_stats
        ),

        "temporal_range": {
            "min_year": years[0],
            "max_year": years[-1],
            "n_observed_years": len(years),
        },

        "candidate_snapshot_pairs": [
            clean_result(r)
            for r in results
        ],

        "selection_rule": (
            "Maximize min(#Added, #Unchanged). "
            "If tied, prefer the smaller temporal gap. "
            "Attack performance is not used to select snapshots."
        ),

        "selected": (
            clean_result(
                best
            )
        ),
    }

    with open(
        output_dir / "split_info.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Saved to: {output_dir}"
    )

    return {
        "dataset": dataset,
        "t1": best["t1"],
        "t2": best["t2"],
        "G1": best["G1_size"],
        "G2": best["G2_size"],
        "Added": best["Added"],
        "Unchanged": best["Unchanged"],
        "Removed_report_only": (
            best["Removed_report_only"]
        ),
        "balanced_capacity": (
            best["balanced_capacity"]
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    summaries = []

    for dataset, path in DATASETS.items():

        summary = process_dataset(
            dataset,
            path
        )

        summaries.append(
            summary
        )

    # ========================================================
    # SAVE INTERVAL DATASET SUMMARY
    # ========================================================

    summary_path = (
        OUTPUT_ROOT
        /
        "interval_dkg_split_summary.csv"
    )

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

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL INTERVAL-DKG SUMMARY")
    print("=" * 100)

    for s in summaries:

        print(
            f"{s['dataset']:15s} | "
            f"t1={s['t1']} | "
            f"t2={s['t2']} | "
            f"G1={s['G1']:,} | "
            f"G2={s['G2']:,} | "
            f"Added={s['Added']:,} | "
            f"Unchanged={s['Unchanged']:,} | "
            f"capacity={s['balanced_capacity']:,}"
        )

    print()
    print(
        f"Summary saved to:"
    )

    print(
        summary_path
    )

    print()
    print(
        "ICEWS05-15 and ICEWS14 were NOT modified."
    )


if __name__ == "__main__":
    main()