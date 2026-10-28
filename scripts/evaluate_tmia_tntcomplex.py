#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR=Path('/home/harrouch/Paper2/Paper2')
ATTACK_ROOT=BASE_DIR/'attack_data_tntcomplex'
RESULTS_ROOT=BASE_DIR/'results'
DATASETS=['ICEWS05-15','ICEWS14','YAGO11k','WIKIDATA12k']
SEED=42
BASELINE_FEATURES=['score_t']
TEMPORAL_FEATURES=['score_t_minus_2','score_t_minus_1','score_t','score_t_plus_1','score_t_plus_2']
RESULTS_ROOT.mkdir(parents=True,exist_ok=True)

def clf():
    return Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(max_iter=1000,random_state=SEED))])

def metrics(y,prob):
    pred=(prob>=0.5).astype(int)
    return {
        'accuracy':accuracy_score(y,pred),
        'precision':precision_score(y,pred,zero_division=0),
        'recall':recall_score(y,pred,zero_division=0),
        'f1':f1_score(y,pred,zero_division=0),
        'roc_auc':roc_auc_score(y,prob),
    }

def evaluate_dataset(ds):
    d=ATTACK_ROOT/ds
    shadow=pd.read_csv(d/'shadow_attack_dataset.csv')
    victim=pd.read_csv(d/'victim_attack_dataset.csv')
    # Common full-window filter for BOTH baseline and T-MIA.
    shadow_clean=shadow.dropna(subset=TEMPORAL_FEATURES).reset_index(drop=True)
    victim_clean=victim.dropna(subset=TEMPORAL_FEATURES).reset_index(drop=True)
    print('\n'+'='*100); print(f'TNTComplEx STANDARD MIA | {ds}'); print('='*100)
    print(f'Shadow: {len(shadow)} raw -> {len(shadow_clean)} clean')
    print(f'Victim: {len(victim)} raw -> {len(victim_clean)} clean')
    ys=shadow_clean['label'].astype(int).to_numpy(); yv=victim_clean['label'].astype(int).to_numpy()
    base=clf(); base.fit(shadow_clean[BASELINE_FEATURES].to_numpy(),ys)
    pbase=base.predict_proba(victim_clean[BASELINE_FEATURES].to_numpy())[:,1]
    mb=metrics(yv,pbase)
    temp=clf(); temp.fit(shadow_clean[TEMPORAL_FEATURES].to_numpy(),ys)
    ptemp=temp.predict_proba(victim_clean[TEMPORAL_FEATURES].to_numpy())[:,1]
    mt=metrics(yv,ptemp)
    gain=mt['roc_auc']-mb['roc_auc']
    print(f"Baseline ROC-AUC: {mb['roc_auc']:.6f}")
    print(f"T-MIA ROC-AUC   : {mt['roc_auc']:.6f}")
    print(f"AUC gain        : {gain:+.6f}")
    return {
        'dataset':ds,
        'shadow_raw_samples':len(shadow),'shadow_clean_samples':len(shadow_clean),
        'victim_raw_samples':len(victim),'victim_clean_samples':len(victim_clean),
        'baseline_accuracy':mb['accuracy'],'baseline_precision':mb['precision'],'baseline_recall':mb['recall'],'baseline_f1':mb['f1'],'baseline_roc_auc':mb['roc_auc'],
        'temporal_accuracy':mt['accuracy'],'temporal_precision':mt['precision'],'temporal_recall':mt['recall'],'temporal_f1':mt['f1'],'temporal_roc_auc':mt['roc_auc'],
        'auc_gain':gain,
    }

if __name__=='__main__':
    rows=[evaluate_dataset(ds) for ds in DATASETS]
    df=pd.DataFrame(rows)
    out=RESULTS_ROOT/'tntcomplex_standard_tmia_results.csv'; df.to_csv(out,index=False)
    print('\n'+'='*100); print('FINAL TNTComplEx STANDARD MIA / T-MIA RESULTS'); print('='*100)
    print(df[['dataset','baseline_roc_auc','temporal_roc_auc','auc_gain','shadow_clean_samples','victim_clean_samples']].to_string(index=False))
    print(f'\nSaved: {out}')
