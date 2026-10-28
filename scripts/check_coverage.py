from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

PROCESSED_ROOT = Path("/home/harrouche/Paper2/processed")

DATASETS = [
    "ICEWS05-15",
    "ICEWS14",
    "YAGO11k",
    "WIKIDATA12k",
]


# ============================================================
# LOAD FACTS
# ============================================================

def load_facts(path):
    """
    Load processed quadruples:
        h r o t

    Returns:
        list of tuples (h, r, o, t)
    """

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

            facts.append((h, r, o, t))

    return facts


# ============================================================
# COVERAGE
# ============================================================

def check_coverage(train_facts, test_facts):

    # --------------------------------------------------------
    # Elements observed during model training
    # --------------------------------------------------------

    train_heads = {h for h, r, o, t in train_facts}
    train_objects = {o for h, r, o, t in train_facts}

    # Entity embeddings are shared between head/object roles,
    # so an entity is considered seen if it appeared anywhere
    # as h OR o during training.
    train_entities = train_heads | train_objects

    train_relations = {
        r for h, r, o, t in train_facts
    }

    train_times = {
        t for h, r, o, t in train_facts
    }

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total = len(test_facts)

    h_seen = 0
    r_seen = 0
    o_seen = 0
    t_seen = 0

    hro_seen = 0
    hrot_seen = 0

    unseen_heads = set()
    unseen_objects = set()
    unseen_relations = set()
    unseen_times = set()

    # --------------------------------------------------------
    # Check every test fact
    # --------------------------------------------------------

    for h, r, o, t in test_facts:

        hs = h in train_entities
        rs = r in train_relations
        os = o in train_entities
        ts = t in train_times

        if hs:
            h_seen += 1
        else:
            unseen_heads.add(h)

        if rs:
            r_seen += 1
        else:
            unseen_relations.add(r)

        if os:
            o_seen += 1
        else:
            unseen_objects.add(o)

        if ts:
            t_seen += 1
        else:
            unseen_times.add(t)

        if hs and rs and os:
            hro_seen += 1

        if hs and rs and os and ts:
            hrot_seen += 1

    # --------------------------------------------------------
    # Safe percentage
    # --------------------------------------------------------

    def pct(x):
        if total == 0:
            return 0.0

        return 100.0 * x / total

    return {
        "total": total,

        "h_seen": h_seen,
        "r_seen": r_seen,
        "o_seen": o_seen,
        "t_seen": t_seen,

        "hro_seen": hro_seen,
        "hrot_seen": hrot_seen,

        "h_seen_pct": pct(h_seen),
        "r_seen_pct": pct(r_seen),
        "o_seen_pct": pct(o_seen),
        "t_seen_pct": pct(t_seen),

        "hro_seen_pct": pct(hro_seen),
        "hrot_seen_pct": pct(hrot_seen),

        "unseen_heads": len(unseen_heads),
        "unseen_objects": len(unseen_objects),
        "unseen_relations": len(unseen_relations),
        "unseen_times": len(unseen_times),
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_coverage(title, results):

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

    print(
        f"Total test facts             : "
        f"{results['total']}"
    )

    print()

    print(
        f"Head entity seen             : "
        f"{results['h_seen_pct']:.2f}% "
        f"({results['h_seen']}/{results['total']})"
    )

    print(
        f"Relation seen                : "
        f"{results['r_seen_pct']:.2f}% "
        f"({results['r_seen']}/{results['total']})"
    )

    print(
        f"Object entity seen           : "
        f"{results['o_seen_pct']:.2f}% "
        f"({results['o_seen']}/{results['total']})"
    )

    print(
        f"Timestamp seen               : "
        f"{results['t_seen_pct']:.2f}% "
        f"({results['t_seen']}/{results['total']})"
    )

    print()

    print(
        f"ALL h/r/o seen               : "
        f"{results['hro_seen_pct']:.2f}% "
        f"({results['hro_seen']}/{results['total']})"
    )

    print(
        f"ALL h/r/o/t seen             : "
        f"{results['hrot_seen_pct']:.2f}% "
        f"({results['hrot_seen']}/{results['total']})"
    )

    print()

    print(
        f"Unique unseen head entities  : "
        f"{results['unseen_heads']}"
    )

    print(
        f"Unique unseen object entities: "
        f"{results['unseen_objects']}"
    )

    print(
        f"Unique unseen relations      : "
        f"{results['unseen_relations']}"
    )

    print(
        f"Unique unseen timestamps     : "
        f"{results['unseen_times']}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ENTITY / RELATION / TIMESTAMP COVERAGE CHECK")
    print("=" * 70)

    for dataset in DATASETS:

        print("\n\n" + "=" * 70)
        print(dataset)
        print("=" * 70)

        dataset_dir = PROCESSED_ROOT / dataset

        # ====================================================
        # VICTIM
        # ====================================================

        victim_train = load_facts(
            dataset_dir / "victim_train.txt"
        )

        victim_test = load_facts(
            dataset_dir / "victim_test.txt"
        )

        victim_results = check_coverage(
            victim_train,
            victim_test
        )

        print_coverage(
            "VICTIM: victim_test vs victim_train",
            victim_results
        )

        # ====================================================
        # SHADOW
        # ====================================================

        shadow_train = load_facts(
            dataset_dir / "shadow_train.txt"
        )

        shadow_test = load_facts(
            dataset_dir / "shadow_test.txt"
        )

        shadow_results = check_coverage(
            shadow_train,
            shadow_test
        )

        print_coverage(
            "SHADOW: shadow_test vs shadow_train",
            shadow_results
        )

    print("\n\n" + "=" * 70)
    print("COVERAGE CHECK FINISHED")
    print("=" * 70)