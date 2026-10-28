#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR=Path('/home/harrouch/Paper2/Paper2')
ATTACK_ROOT=BASE_DIR/'attack_data_tcomplex'
RESULTS_DIR=BASE_DIR/'results'; RESULTS_DIR.mkdir(parents=True,exist_ok=True)
DATASETS=['ICEWS05-15','ICEWS14','YAGO11k','WIKIDATA12k']
BASELINE=['score_t']
TEMPORAL=['score_t_minus_2','score_t_minus_1','score_t','score_t_plus_1','score_t_plus_2']

def run_attack(shadow,victim,features,name):
    Xs=shadow[features].values; ys=shadow['label'].values
    Xv=victim[features].values; yv=victim['label'].values
    clf=Pipeline([('scaler',StandardScaler()),('lr',LogisticRegression(max_iter=1000,random_state=42))])
    clf.fit(Xs,ys)
    pred=clf.predict(Xv); prob=clf.predict_proba(Xv)[:,1]
    auc=roc_auc_score(yv,prob)
    return {'attack':name,
            'accuracy':accuracy_score(yv,pred),
            'precision':precision_score(yv,pred,zero_division=0),
            'recall':recall_score(yv,pred,zero_division=0),
            'f1':f1_score(yv,pred,zero_division=0),
            'roc_auc':auc,
            'effective_auc':max(auc,1-auc),
            'attack_advantage':abs(2*auc-1),
            'shadow_samples':len(shadow),'victim_samples':len(victim)}

results=[]
print('='*90); print('TCOMPLEX BASELINE-MIA / TEMPORAL-MIA'); print('='*90)
for dataset in DATASETS:
    print('\n'+'='*90); print('DATASET:',dataset); print('='*90)
    d=ATTACK_ROOT/dataset
    shadow=pd.read_csv(d/'shadow_attack_dataset.csv')
    victim=pd.read_csv(d/'victim_attack_dataset.csv')
    # Same complete-window examples for baseline and temporal attacks.
    shadow=shadow.dropna(subset=TEMPORAL).reset_index(drop=True)
    victim=victim.dropna(subset=TEMPORAL).reset_index(drop=True)
    print('Clean shadow samples :',len(shadow)); print('Clean victim samples :',len(victim))
    b=run_attack(shadow,victim,BASELINE,'Baseline-MIA')
    t=run_attack(shadow,victim,TEMPORAL,'Temporal-MIA')
    b['dataset']=dataset; t['dataset']=dataset; results.extend([b,t])
    print(f"Baseline ROC-AUC : {b['roc_auc']:.6f}")
    print(f"Temporal ROC-AUC : {t['roc_auc']:.6f}")
    print(f"Temporal AUC gain: {t['roc_auc']-b['roc_auc']:+.6f}")

cols=['dataset','attack','accuracy','precision','recall','f1','roc_auc','effective_auc','attack_advantage','shadow_samples','victim_samples']
df=pd.DataFrame(results)[cols]
print('\n'+'='*90); print('FINAL TCOMPLEX MIA / T-MIA RESULTS'); print('='*90)
print(df.to_string(index=False))
out=RESULTS_DIR/'tmia_tcomplex_results.csv'; df.to_csv(out,index=False)
print('\nSaved:',out)
