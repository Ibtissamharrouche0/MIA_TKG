#!/usr/bin/env python3
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = Path("/home/harrouch/Paper2/Paper2")
PROCESSED_ROOT = BASE_DIR / "processed"
ATTACK_ROOT = BASE_DIR / "attack_data_tcomplex"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["ICEWS05-15", "ICEWS14", "YAGO11k", "WIKIDATA12k"]
ROLES = ["shadow", "victim"]

def load_facts(path):
    facts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{path}, line {i}: expected 4 columns")
            facts.append(tuple(map(int, parts)))
    return facts

def safe_auc(y, scores):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    y, scores = y[mask], scores[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, scores)

def cohen_d(x1, x0):
    x1 = np.asarray(x1, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    x1 = x1[np.isfinite(x1)]
    x0 = x0[np.isfinite(x0)]
    if len(x1) < 2 or len(x0) < 2:
        return float("nan")
    v1 = np.var(x1, ddof=1)
    v0 = np.var(x0, ddof=1)
    pooled = math.sqrt(((len(x1)-1)*v1 + (len(x0)-1)*v0) / (len(x1)+len(x0)-2))
    if pooled == 0:
        return float("nan")
    return (np.mean(x1) - np.mean(x0)) / pooled

def summarize_scores(df):
    members = df.loc[df["label"] == 1, "score_t"].dropna().to_numpy()
    nonmembers = df.loc[df["label"] == 0, "score_t"].dropna().to_numpy()
    y = df["label"].to_numpy()
    s = df["score_t"].to_numpy()
    auc_pos = safe_auc(y, s)
    auc_neg = safe_auc(y, -s)
    return {
        "member_mean": np.mean(members),
        "member_std": np.std(members),
        "member_median": np.median(members),
        "nonmember_mean": np.mean(nonmembers),
        "nonmember_std": np.std(nonmembers),
        "nonmember_median": np.median(nonmembers),
        "direct_score_auc": auc_pos,
        "inverse_score_auc": auc_neg,
        "best_direction_auc": max(auc_pos, auc_neg),
        "cohen_d": cohen_d(members, nonmembers),
    }

def build_train_sets(train_facts):
    entities, relations, times, triples, quads = set(), set(), set(), set(), set()
    for h, r, o, t in train_facts:
        entities.update([h, o])
        relations.add(r)
        times.add(t)
        triples.add((h, r, o))
        quads.add((h, r, o, t))
    return {
        "entities": entities,
        "relations": relations,
        "times": times,
        "triples": triples,
        "quads": quads,
    }

def coverage_stats(nonmember_facts, train_sets):
    n = len(nonmember_facts)
    counts = {
        "head": 0, "tail": 0, "both": 0, "rel": 0, "time": 0,
        "all": 0, "triple": 0, "quad": 0
    }
    for h, r, o, t in nonmember_facts:
        hs = h in train_sets["entities"]
        os = o in train_sets["entities"]
        rs = r in train_sets["relations"]
        ts = t in train_sets["times"]
        counts["head"] += int(hs)
        counts["tail"] += int(os)
        counts["both"] += int(hs and os)
        counts["rel"] += int(rs)
        counts["time"] += int(ts)
        counts["all"] += int(hs and os and rs and ts)
        counts["triple"] += int((h, r, o) in train_sets["triples"])
        counts["quad"] += int((h, r, o, t) in train_sets["quads"])
    pct = lambda x: 100.0 * x / n if n else float("nan")
    return {
        "nm_head_seen_pct": pct(counts["head"]),
        "nm_tail_seen_pct": pct(counts["tail"]),
        "nm_both_entities_seen_pct": pct(counts["both"]),
        "nm_relation_seen_pct": pct(counts["rel"]),
        "nm_time_seen_pct": pct(counts["time"]),
        "nm_all_vocab_seen_pct": pct(counts["all"]),
        "nm_triple_overlap_pct": pct(counts["triple"]),
        "nm_quad_overlap_pct": pct(counts["quad"]),
    }

def tag_seen(df, train_sets):
    out = df.copy()
    out["all_vocab_seen"] = out.apply(
        lambda row:
            int(row["h"]) in train_sets["entities"]
            and int(row["o"]) in train_sets["entities"]
            and int(row["r"]) in train_sets["relations"]
            and int(row["t"]) in train_sets["times"],
        axis=1
    )
    return out

def subset_auc(df, mask):
    sub = df.loc[mask].copy()
    if len(sub) == 0 or sub["label"].nunique() < 2:
        return len(sub), float("nan")
    return len(sub), safe_auc(sub["label"], sub["score_t"])

rows = []

print("=" * 100)
print("TCOMPLEX MIA DIAGNOSTIC")
print("=" * 100)

for dataset in DATASETS:
    for role in ROLES:
        print("\n" + "=" * 100)
        print(f"{dataset} | {role.upper()}")
        print("=" * 100)

        dataset_dir = PROCESSED_ROOT / dataset
        attack_path = ATTACK_ROOT / dataset / f"{role}_attack_dataset.csv"
        train_path = dataset_dir / f"{role}_train.txt"
        test_path = dataset_dir / f"{role}_test.txt"

        if not attack_path.exists():
            raise FileNotFoundError(attack_path)

        attack_df = pd.read_csv(attack_path)
        train_facts = load_facts(train_path)
        test_facts = load_facts(test_path)

        train_sets = build_train_sets(train_facts)
        score_stats = summarize_scores(attack_df)
        cover = coverage_stats(test_facts, train_sets)
        tagged = tag_seen(attack_df, train_sets)

        seen_n, seen_auc = subset_auc(tagged, tagged["all_vocab_seen"])
        unseen_n, unseen_auc = subset_auc(tagged, ~tagged["all_vocab_seen"])

        print(f"Attack samples       : {len(attack_df)}")
        print(f"Train facts          : {len(train_facts)}")
        print(f"Test/nonmember facts : {len(test_facts)}")

        print("\nScore distribution")
        print(f"  Member mean        : {score_stats['member_mean']:.6f}")
        print(f"  Non-member mean    : {score_stats['nonmember_mean']:.6f}")
        print(f"  Member median      : {score_stats['member_median']:.6f}")
        print(f"  Non-member median  : {score_stats['nonmember_median']:.6f}")
        print(f"  Cohen d            : {score_stats['cohen_d']:.6f}")

        print("\nDirect score diagnostic")
        print(f"  AUC(score_t)       : {score_stats['direct_score_auc']:.6f}")
        print(f"  AUC(-score_t)      : {score_stats['inverse_score_auc']:.6f}")
        print(f"  Best direction AUC : {score_stats['best_direction_auc']:.6f}")

        print("\nNon-member coverage in training vocabulary")
        print(f"  Head seen          : {cover['nm_head_seen_pct']:.2f}%")
        print(f"  Tail seen          : {cover['nm_tail_seen_pct']:.2f}%")
        print(f"  Both entities seen : {cover['nm_both_entities_seen_pct']:.2f}%")
        print(f"  Relation seen      : {cover['nm_relation_seen_pct']:.2f}%")
        print(f"  Time seen          : {cover['nm_time_seen_pct']:.2f}%")
        print(f"  All vocab seen     : {cover['nm_all_vocab_seen_pct']:.2f}%")
        print(f"  Triple overlap     : {cover['nm_triple_overlap_pct']:.2f}%")
        print(f"  Quad overlap       : {cover['nm_quad_overlap_pct']:.2f}%")

        print("\nAUC by vocabulary visibility")
        print(f"  all-vocab-seen samples : {seen_n}")
        print(f"  all-vocab-seen AUC     : {seen_auc:.6f}" if math.isfinite(seen_auc) else "  all-vocab-seen AUC     : NaN")
        print(f"  some-unseen samples    : {unseen_n}")
        print(f"  some-unseen AUC        : {unseen_auc:.6f}" if math.isfinite(unseen_auc) else "  some-unseen AUC        : NaN")

        rows.append({
            "dataset": dataset,
            "role": role,
            "attack_samples": len(attack_df),
            "train_facts": len(train_facts),
            "test_facts": len(test_facts),
            **score_stats,
            **cover,
            "all_vocab_seen_attack_samples": seen_n,
            "all_vocab_seen_score_auc": seen_auc,
            "some_unseen_attack_samples": unseen_n,
            "some_unseen_score_auc": unseen_auc,
        })

summary = pd.DataFrame(rows)

output_path = RESULTS_DIR / "tcomplex_mia_diagnostic_summary.csv"
summary.to_csv(output_path, index=False)

print("\n" + "=" * 100)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 100)

cols = [
    "dataset",
    "role",
    "direct_score_auc",
    "best_direction_auc",
    "cohen_d",
    "nm_all_vocab_seen_pct",
    "nm_triple_overlap_pct",
    "nm_quad_overlap_pct",
    "all_vocab_seen_score_auc",
    "some_unseen_score_auc",
]

print(summary[cols].to_string(index=False))
print("\nSaved:")
print(output_path)
