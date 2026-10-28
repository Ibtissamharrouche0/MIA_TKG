import json
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PRIVACY_ROOT = Path(
    "/home/harrouche/Paper2/privacy_splits"
)

OUTPUT_ROOT = Path(
    "/home/harrouche/Paper2/processed"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


DATASETS = {
    "ICEWS05-15": {
        "type": "point"
    },

    "ICEWS14": {
        "type": "point"
    },

    "YAGO11k": {
        "type": "interval"
    },

    "WIKIDATA12k": {
        "type": "interval"
    }
}


SPLIT_FILES = {
    "victim_train": "victim_train_raw.txt",
    "victim_test": "victim_test_raw.txt",
    "shadow_train": "shadow_train_raw.txt",
    "shadow_test": "shadow_test_raw.txt",
}


# ============================================================
# TEMPORAL HELPERS
# ============================================================

def extract_year(value):
    """
    Examples:
        1990-##-##  -> 1990
        1990-05-##  -> 1990
        1990-05-10  -> 1990
        ####-##-##  -> None
    """

    value = value.strip()

    if value.startswith("####"):
        return None

    year_part = value.split("-")[0]

    try:
        return int(year_part)

    except ValueError:
        return None


# ============================================================
# LOAD RAW SPLIT
# ============================================================

def load_lines(path):

    lines = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                lines.append(line)

    return lines


# ============================================================
# PROCESS ICEWS
# ============================================================

def preprocess_point_lines(lines):

    facts = []

    timestamps = set()

    interval_count = 0

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        parts = line.split()

        if len(parts) == 5:

            h = int(parts[0])
            r = int(parts[1])
            o = int(parts[2])

            start_time = parts[3]
            end_time = parts[4]

            if start_time != end_time:
                interval_count += 1

            timestamp = start_time

        elif len(parts) == 4:

            h = int(parts[0])
            r = int(parts[1])
            o = int(parts[2])

            timestamp = parts[3]

        else:

            raise ValueError(
                f"Unsupported point format at line "
                f"{line_number}: {line}"
            )

        facts.append(
            (
                h,
                r,
                o,
                timestamp
            )
        )

        timestamps.add(timestamp)

    return facts, timestamps, {
        "interval_count": interval_count
    }


# ============================================================
# PROCESS YAGO / WIKIDATA
# ============================================================

def preprocess_interval_lines(lines):

    facts = []

    years = set()

    stats = {
        "full_intervals": 0,
        "partial_intervals": 0,
        "single_year": 0,
        "skipped": 0,
    }

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        parts = line.split()

        # ----------------------------------------------------
        # 4 columns: h r o time
        # ----------------------------------------------------

        if len(parts) == 4:

            h = int(parts[0])
            r = int(parts[1])
            o = int(parts[2])

            raw_time = parts[3]

            year = extract_year(
                raw_time
            )

            if year is None:

                stats["skipped"] += 1
                continue

            facts.append(
                (
                    h,
                    r,
                    o,
                    year
                )
            )

            years.add(year)

            stats["single_year"] += 1

        # ----------------------------------------------------
        # 5 columns: h r o start end
        # ----------------------------------------------------

        elif len(parts) == 5:

            h = int(parts[0])
            r = int(parts[1])
            o = int(parts[2])

            start_raw = parts[3]
            end_raw = parts[4]

            start_year = extract_year(
                start_raw
            )

            end_year = extract_year(
                end_raw
            )

            # both unknown
            if (
                start_year is None
                and
                end_year is None
            ):

                stats["skipped"] += 1
                continue

            # unknown start
            if (
                start_year is None
                and
                end_year is not None
            ):

                facts.append(
                    (
                        h,
                        r,
                        o,
                        end_year
                    )
                )

                years.add(end_year)

                stats["partial_intervals"] += 1
                continue

            # unknown end
            if (
                start_year is not None
                and
                end_year is None
            ):

                facts.append(
                    (
                        h,
                        r,
                        o,
                        start_year
                    )
                )

                years.add(start_year)

                stats["partial_intervals"] += 1
                continue

            # same year
            if start_year == end_year:

                facts.append(
                    (
                        h,
                        r,
                        o,
                        start_year
                    )
                )

                years.add(start_year)

                stats["single_year"] += 1
                continue

            # valid interval
            if start_year < end_year:

                for year in range(
                    start_year,
                    end_year + 1
                ):

                    facts.append(
                        (
                            h,
                            r,
                            o,
                            year
                        )
                    )

                    years.add(year)

                stats["full_intervals"] += 1
                continue

            # invalid interval
            stats["skipped"] += 1

        else:

            raise ValueError(
                f"Unsupported interval format at line "
                f"{line_number}: {line}"
            )

    # Remove duplicates generated by interval expansion
    facts = list(
        set(facts)
    )

    return facts, years, stats


# ============================================================
# CONVERT TEMPORAL VALUES TO IDS
# ============================================================

def convert_point_to_ids(
    facts,
    time2id
):

    converted = []

    for h, r, o, timestamp in facts:

        converted.append(
            (
                h,
                r,
                o,
                time2id[timestamp]
            )
        )

    return converted


def convert_interval_to_ids(
    facts,
    time2id
):

    converted = []

    for h, r, o, year in facts:

        converted.append(
            (
                h,
                r,
                o,
                time2id[str(year)]
            )
        )

    return converted


# ============================================================
# SAVE
# ============================================================

def save_split(
    path,
    facts
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for h, r, o, t in facts:

            f.write(
                f"{h}\t{r}\t{o}\t{t}\n"
            )


# ============================================================
# DATASET PROCESSING
# ============================================================

def process_dataset(
    dataset_name,
    dataset_type
):

    print("\n" + "=" * 70)
    print(f"PREPROCESSING {dataset_name}")
    print("=" * 70)

    input_dir = (
        PRIVACY_ROOT
        /
        dataset_name
    )

    output_dir = (
        OUTPUT_ROOT
        /
        dataset_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_split_data = {}
    processed_split_data = {}

    global_times = set()

    split_stats = {}

    # ========================================================
    # STEP 1 — READ AND PREPROCESS EACH RAW SPLIT
    # ========================================================

    for split_name, filename in SPLIT_FILES.items():

        path = (
            input_dir
            /
            filename
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing split: {path}"
            )

        lines = load_lines(
            path
        )

        print(
            f"\n{split_name}: {len(lines)} raw facts"
        )

        if dataset_type == "point":

            facts, times, stats = preprocess_point_lines(
                lines
            )

        elif dataset_type == "interval":

            facts, times, stats = preprocess_interval_lines(
                lines
            )

        else:

            raise ValueError(
                f"Unknown dataset type: {dataset_type}"
            )

        raw_split_data[
            split_name
        ] = facts

        split_stats[
            split_name
        ] = stats

        global_times.update(
            times
        )

    # ========================================================
    # STEP 2 — BUILD ONE GLOBAL TIME MAPPING
    # ========================================================

    if dataset_type == "point":

        sorted_times = sorted(
            global_times
        )

        time2id = {
            timestamp: idx
            for idx, timestamp
            in enumerate(sorted_times)
        }

    else:

        sorted_times = sorted(
            global_times
        )

        time2id = {
            str(year): idx
            for idx, year
            in enumerate(sorted_times)
        }

    print(
        f"\nGlobal temporal vocabulary: "
        f"{len(time2id)} timestamps"
    )

    # ========================================================
    # STEP 3 — CONVERT EACH SPLIT
    # ========================================================

    for split_name, facts in raw_split_data.items():

        if dataset_type == "point":

            converted = convert_point_to_ids(
                facts,
                time2id
            )

        else:

            converted = convert_interval_to_ids(
                facts,
                time2id
            )

        converted = sorted(
            converted
        )

        processed_split_data[
            split_name
        ] = converted

    # ========================================================
    # STEP 4 — SAVE
    # ========================================================

    for split_name, facts in processed_split_data.items():

        output_path = (
            output_dir
            /
            f"{split_name}.txt"
        )

        save_split(
            output_path,
            facts
        )

    # Save temporal mapping
    mapping_path = (
        output_dir
        /
        "time2id.json"
    )

    with open(
        mapping_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            time2id,
            f,
            indent=4
        )

    # ========================================================
    # STEP 5 — VERIFY VICTIM / SHADOW DISJOINTNESS
    # ========================================================

    victim_side = (
        set(processed_split_data["victim_train"])
        |
        set(processed_split_data["victim_test"])
    )

    shadow_side = (
        set(processed_split_data["shadow_train"])
        |
        set(processed_split_data["shadow_test"])
    )

    overlap = (
        victim_side
        &
        shadow_side
    )

    print("\n--------------------------------------------")

    if overlap:

        print(
            f"WARNING: {len(overlap)} processed quadruples "
            f"appear on both victim and shadow sides."
        )

        print(
            "This may happen when different original temporal "
            "intervals expand to the same quadruple."
        )

    else:

        print(
            "Victim / shadow processed overlap: 0 ✓"
        )

    # ========================================================
    # STEP 6 — REPORT
    # ========================================================

    for split_name in SPLIT_FILES:

        print(
            f"{split_name:15s}: "
            f"{len(processed_split_data[split_name])}"
        )

        stats = split_stats[
            split_name
        ]

        if dataset_type == "interval":

            print(
                f"    full intervals    : "
                f"{stats['full_intervals']}"
            )

            print(
                f"    partial intervals : "
                f"{stats['partial_intervals']}"
            )

            print(
                f"    single-year       : "
                f"{stats['single_year']}"
            )

            print(
                f"    skipped           : "
                f"{stats['skipped']}"
            )

    print(
        f"\ntime2id: {mapping_path}"
    )

    print(
        f"Output : {output_dir}"
    )

    print("--------------------------------------------")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TKG PRIVACY-SPLIT TEMPORAL PREPROCESSING")
    print("=" * 70)

    successful = []
    failed = []

    for dataset_name, config in DATASETS.items():

        try:

            process_dataset(
                dataset_name,
                config["type"]
            )

            successful.append(
                dataset_name
            )

        except Exception as e:

            print(
                f"\nERROR processing "
                f"{dataset_name}"
            )

            print(
                e
            )

            failed.append(
                dataset_name
            )

    print("\n" + "=" * 70)
    print("PREPROCESSING FINISHED")
    print("=" * 70)

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

    print(
        f"\nProcessed root: {OUTPUT_ROOT}"
    )