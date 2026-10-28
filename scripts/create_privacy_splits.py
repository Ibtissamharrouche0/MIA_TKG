import random
from collections import defaultdict
from pathlib import Path


SEED = 42

DATASETS = {
    "ICEWS05-15": "/home/harrouche/Paper2/data/icews05-15/train.txt",
    "ICEWS14": "/home/harrouche/Paper2/data/icews14/train.txt",
    "YAGO11k": "/home/harrouche/Paper2/data/YAGO11k/train.txt",
    "WIKIDATA12k": "/home/harrouche/Paper2/data/WIKIDATA12k/train.txt",
}

OUTPUT_ROOT = Path(
    "/home/harrouche/Paper2/privacy_splits"
)

RATIOS = {
    "victim_train": 0.40,
    "victim_test": 0.10,
    "shadow_train": 0.40,
    "shadow_test": 0.10,
}


def load_grouped_facts(path):

    groups = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            h = parts[0]
            r = parts[1]
            o = parts[2]

            key = (h, r, o)

            groups[key].append(line)

    return groups


def save_split(path, facts):

    with open(path, "w", encoding="utf-8") as f:

        for fact in facts:
            f.write(fact + "\n")


def split_dataset(dataset_name, input_path):

    print("\n" + "=" * 65)
    print(f"CREATING GROUPED PRIVACY SPLITS FOR {dataset_name}")
    print("=" * 65)

    groups = load_grouped_facts(input_path)

    group_items = list(groups.items())

    rng = random.Random(SEED)
    rng.shuffle(group_items)

    total_facts = sum(
        len(facts)
        for _, facts in group_items
    )

    targets = {
        name: total_facts * ratio
        for name, ratio in RATIOS.items()
    }

    splits = {
        "victim_train": [],
        "victim_test": [],
        "shadow_train": [],
        "shadow_test": [],
    }

    counts = {
        name: 0
        for name in splits
    }

    # --------------------------------------------------------
    # Assign complete (h,r,o) groups
    # --------------------------------------------------------

    for triple, facts in group_items:

        # Select partition that is furthest below its target
        best_split = max(
            splits.keys(),
            key=lambda name:
                targets[name] - counts[name]
        )

        splits[best_split].extend(facts)

        counts[best_split] += len(facts)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = OUTPUT_ROOT / dataset_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for split_name, facts in splits.items():

        save_split(
            output_dir / f"{split_name}_raw.txt",
            facts
        )

    # --------------------------------------------------------
    # Verify triples are disjoint
    # --------------------------------------------------------

    triple_sets = {}

    for split_name, facts in splits.items():

        triple_sets[split_name] = {
            tuple(line.split()[:3])
            for line in facts
        }

    names = list(triple_sets.keys())

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            overlap = (
                triple_sets[names[i]]
                &
                triple_sets[names[j]]
            )

            if overlap:

                raise RuntimeError(
                    f"Triple overlap between "
                    f"{names[i]} and {names[j]}: "
                    f"{len(overlap)}"
                )

    print(f"Original facts : {total_facts}")
    print(f"Unique triples : {len(groups)}")

    print("--------------------------------------------")

    for name in splits:

        print(
            f"{name:15s}: "
            f"{len(splits[name])}"
        )

    print("Triple overlap : 0 ✓")
    print(f"Output         : {output_dir}")


if __name__ == "__main__":

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for dataset_name, path in DATASETS.items():

        path = Path(path)

        if not path.exists():

            print(f"Missing: {path}")
            continue

        split_dataset(
            dataset_name,
            path
        )

    print("\nPrivacy split creation finished.")