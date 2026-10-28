
# controlled_same_triple_tmia_tntcomplex.py
#
# Same-triple controlled temporal membership inference for TNTComplEx.
#
# IMPORTANT:
# - Does NOT retrain TNTComplEx.
# - Does NOT create new controlled pairs.
# - Reuses EXACTLY the same pair CSVs already used for TComplEx/TeRo-like:
#     attack_data_tcomplex_same_triple/<DATASET>/shadow_same_triple_pairs.csv
#     attack_data_tcomplex_same_triple/<DATASET>/victim_same_triple_pairs.csv
# - Member and synthetic non-member keep the same (h,r,o); only t differs.
# - Baseline = [s(t)]
# - T-MIA    = [s(t-2),s(t-1),s(t),s(t+1),s(t+2)]
# - Common full-window filtering for a fair comparison.
# - LR is trained on SHADOW and evaluated on VICTIM.

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

BASE_DIR = Path("/home/harrouch/Paper2/Paper2")
PROCESSED_ROOT = BASE_DIR / "processed"
MODEL_ROOT = BASE_DIR / "models"

# Reuse the SAME controlled pairs from the previous controlled experiments.
PAIR_ROOT = BASE_DIR / "attack_data_tcomplex_same_triple"

# TNTComplEx-specific scored controlled attack datasets.
OUTPUT_ROOT = BASE_DIR / "attack_data_tntcomplex_same_triple"
RESULTS_ROOT = BASE_DIR / "results"

DATASETS = ["ICEWS05-15", "ICEWS14", "YAGO11k", "WIKIDATA12k"]
SEED = 42

BASELINE_FEATURES = ["score_t"]
TEMPORAL_FEATURES = [
    "score_t_minus_2", "score_t_minus_1", "score_t",
    "score_t_plus_1", "score_t_plus_2"
]

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def complex_mul(a_re, a_im, b_re, b_im):
    return a_re*b_re - a_im*b_im, a_re*b_im + a_im*b_re


class TNTComplEx(nn.Module):
    def __init__(self, num_entities, num_relations_with_reciprocals,
                 num_times, embedding_dim):
        super().__init__()
        self.ent_re = nn.Embedding(num_entities, embedding_dim)
        self.ent_im = nn.Embedding(num_entities, embedding_dim)

        self.rel_time_re = nn.Embedding(num_relations_with_reciprocals, embedding_dim)
        self.rel_time_im = nn.Embedding(num_relations_with_reciprocals, embedding_dim)

        self.rel_static_re = nn.Embedding(num_relations_with_reciprocals, embedding_dim)
        self.rel_static_im = nn.Embedding(num_relations_with_reciprocals, embedding_dim)

        self.time_re = nn.Embedding(num_times, embedding_dim)
        self.time_im = nn.Embedding(num_times, embedding_dim)

    def score(self, h, r, o, t):
        h_re, h_im = self.ent_re(h), self.ent_im(h)
        rt_re, rt_im = self.rel_time_re(r), self.rel_time_im(r)
        rs_re, rs_im = self.rel_static_re(r), self.rel_static_im(r)
        tau_re, tau_im = self.time_re(t), self.time_im(t)

        dyn_re, dyn_im = complex_mul(rt_re, rt_im, tau_re, tau_im)
        rel_re, rel_im = dyn_re + rs_re, dyn_im + rs_im

        q_re, q_im = complex_mul(h_re, h_im, rel_re, rel_im)
        o_re, o_im = self.ent_re(o), self.ent_im(o)

        # Re(<q, conj(o)>)
        return torch.sum(q_re * o_re + q_im * o_im, dim=1)


def load_model(dataset, role, device):
    path = MODEL_ROOT / dataset / f"tntcomplex_{role}.pt"
    if not path.exists():
        raise FileNotFoundError(path)

    ckpt = torch.load(path, map_location=device)

    model = TNTComplEx(
        ckpt["num_entities"],
        ckpt["num_relations_with_reciprocals"],
        ckpt["num_times"],
        ckpt["embedding_dim"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_time_maps(dataset):
    path = PROCESSED_ROOT / dataset / "time2id.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    time2id = {str(k): int(v) for k, v in raw.items()}
    id2time = {int(v): str(k) for k, v in time2id.items()}
    return time2id, id2time


def shift_real_time(value, delta, dataset):
    if dataset.startswith("ICEWS"):
        dt = pd.Timestamp(value) + pd.Timedelta(days=delta)
        return dt.strftime("%Y-%m-%d")
    return str(int(str(value)) + delta)


def neighbor_id(tid, delta, dataset, time2id, id2time):
    if delta == 0:
        return int(tid)
    if int(tid) not in id2time:
        return None
    shifted = shift_real_time(id2time[int(tid)], delta, dataset)
    return time2id.get(str(shifted))


@torch.no_grad()
def score_batch(model, h, r, o, t, device, batch_size=8192):
    out = []
    n = len(h)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ht = torch.as_tensor(h[start:end], dtype=torch.long, device=device)
        rt = torch.as_tensor(r[start:end], dtype=torch.long, device=device)
        ot = torch.as_tensor(o[start:end], dtype=torch.long, device=device)
        tt = torch.as_tensor(t[start:end], dtype=torch.long, device=device)
        out.append(model.score(ht, rt, ot, tt).detach().cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def detect_columns(df):
    """
    Accept the pair-file naming conventions used by our controlled scripts.
    We need h,r,o plus member and non-member timestamp columns.
    """
    cols = set(df.columns)

    h = next((c for c in ["h", "head", "subject"] if c in cols), None)
    r = next((c for c in ["r", "relation", "predicate"] if c in cols), None)
    o = next((c for c in ["o", "tail", "object"] if c in cols), None)

    tm = next((c for c in [
        "t_member", "member_t", "member_time", "t_m",
        "member_time_id", "member_t_id"
    ] if c in cols), None)

    tn = next((c for c in [
        "t_nonmember", "nonmember_t", "nonmember_time", "t_nm",
        "nonmember_time_id", "nonmember_t_id", "synthetic_t"
    ] if c in cols), None)

    if None in (h, r, o, tm, tn):
        raise ValueError(
            "Could not identify pair columns.\n"
            f"Found columns: {list(df.columns)}\n"
            "Expected h/r/o plus member and non-member timestamp columns."
        )
    return h, r, o, tm, tn


def pairs_to_candidates(pair_df):
    hcol, rcol, ocol, tmcol, tncol = detect_columns(pair_df)

    members = pd.DataFrame({
        "h": pair_df[hcol].astype(int),
        "r": pair_df[rcol].astype(int),
        "o": pair_df[ocol].astype(int),
        "t": pair_df[tmcol].astype(int),
        "label": 1,
        "pair_id": np.arange(len(pair_df)),
    })

    nonmembers = pd.DataFrame({
        "h": pair_df[hcol].astype(int),
        "r": pair_df[rcol].astype(int),
        "o": pair_df[ocol].astype(int),
        "t": pair_df[tncol].astype(int),
        "label": 0,
        "pair_id": np.arange(len(pair_df)),
    })

    # Interleave is unnecessary statistically; deterministic concat is fine.
    return pd.concat([members, nonmembers], ignore_index=True)


def score_controlled_candidates(dataset, role, pair_path, device):
    print("\n" + "="*100)
    print(f"TNTComplEx SAME-TRIPLE CONTROLLED | {dataset} | {role.upper()}")
    print("="*100)
    print(f"Pairs: {pair_path}")

    pairs = pd.read_csv(pair_path)
    candidates = pairs_to_candidates(pairs)

    print(f"Controlled pairs : {len(pairs)}")
    print(f"Attack candidates: {len(candidates)}")

    model = load_model(dataset, role, device)
    time2id, id2time = load_time_maps(dataset)

    h = candidates["h"].to_numpy(dtype=np.int64)
    r = candidates["r"].to_numpy(dtype=np.int64)
    o = candidates["o"].to_numpy(dtype=np.int64)
    t = candidates["t"].to_numpy(dtype=np.int64)

    for delta, col in [
        (-2, "score_t_minus_2"),
        (-1, "score_t_minus_1"),
        (0, "score_t"),
        (1, "score_t_plus_1"),
        (2, "score_t_plus_2"),
    ]:
        candidate_times = np.array([
            neighbor_id(x, delta, dataset, time2id, id2time)
            for x in t
        ], dtype=object)

        valid = np.array([x is not None for x in candidate_times])
        scores = np.full(len(candidates), np.nan, dtype=np.float64)

        if valid.any():
            tids = np.asarray(candidate_times[valid], dtype=np.int64)
            scores[valid] = score_batch(
                model, h[valid], r[valid], o[valid], tids, device
            )

        candidates[col] = scores

    out_dir = OUTPUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{role}_same_triple_attack_dataset.csv"
    candidates.to_csv(out_path, index=False)

    clean = candidates.dropna(subset=TEMPORAL_FEATURES)
    print(f"Full-window rows: {len(clean)}/{len(candidates)}")
    print(f"Saved: {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return candidates


def classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
    }


def evaluate(dataset, shadow_df, victim_df):
    # Same full-window criterion for Baseline and T-MIA.
    shadow = shadow_df.dropna(subset=TEMPORAL_FEATURES).reset_index(drop=True)
    victim = victim_df.dropna(subset=TEMPORAL_FEATURES).reset_index(drop=True)

    ys = shadow["label"].astype(int).to_numpy()
    yv = victim["label"].astype(int).to_numpy()

    base = classifier()
    base.fit(shadow[BASELINE_FEATURES].to_numpy(), ys)
    pbase = base.predict_proba(victim[BASELINE_FEATURES].to_numpy())[:, 1]
    mb = metrics(yv, pbase)

    temp = classifier()
    temp.fit(shadow[TEMPORAL_FEATURES].to_numpy(), ys)
    ptemp = temp.predict_proba(victim[TEMPORAL_FEATURES].to_numpy())[:, 1]
    mt = metrics(yv, ptemp)

    gain = mt["roc_auc"] - mb["roc_auc"]

    print("\nRESULT")
    print(f"Baseline ROC-AUC : {mb['roc_auc']:.6f}")
    print(f"T-MIA ROC-AUC    : {mt['roc_auc']:.6f}")
    print(f"AUC gain         : {gain:+.6f}")
    print(f"Shadow clean     : {len(shadow)}")
    print(f"Victim clean     : {len(victim)}")

    return {
        "dataset": dataset,
        "baseline_accuracy": mb["accuracy"],
        "baseline_precision": mb["precision"],
        "baseline_recall": mb["recall"],
        "baseline_f1": mb["f1"],
        "baseline_roc_auc": mb["roc_auc"],
        "temporal_accuracy": mt["accuracy"],
        "temporal_precision": mt["precision"],
        "temporal_recall": mt["recall"],
        "temporal_f1": mt["f1"],
        "temporal_roc_auc": mt["roc_auc"],
        "auc_gain": gain,
        "shadow_clean_samples": len(shadow),
        "victim_clean_samples": len(victim),
    }


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*100)
    print("TNTComplEx SAME-TRIPLE CONTROLLED MIA / T-MIA")
    print("="*100)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    results = []

    for dataset in DATASETS:
        pair_dir = PAIR_ROOT / dataset
        shadow_pair_path = pair_dir / "shadow_same_triple_pairs.csv"
        victim_pair_path = pair_dir / "victim_same_triple_pairs.csv"

        if not shadow_pair_path.exists():
            raise FileNotFoundError(shadow_pair_path)
        if not victim_pair_path.exists():
            raise FileNotFoundError(victim_pair_path)

        shadow_df = score_controlled_candidates(
            dataset, "shadow", shadow_pair_path, device
        )
        victim_df = score_controlled_candidates(
            dataset, "victim", victim_pair_path, device
        )

        results.append(evaluate(dataset, shadow_df, victim_df))

    results_df = pd.DataFrame(results)
    result_path = RESULTS_ROOT / "tntcomplex_same_triple_tmia_results.csv"
    results_df.to_csv(result_path, index=False)

    print("\n" + "="*100)
    print("FINAL TNTComplEx SAME-TRIPLE CONTROLLED RESULTS")
    print("="*100)
    print(results_df[[
        "dataset", "baseline_roc_auc", "temporal_roc_auc", "auc_gain",
        "shadow_clean_samples", "victim_clean_samples"
    ]].to_string(index=False))
    print(f"\nSaved: {result_path}")
