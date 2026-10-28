#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timedelta
import json, random
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path('/home/harrouch/Paper2/Paper2')
PROCESSED_ROOT = BASE_DIR / 'processed'
MODEL_ROOT = BASE_DIR / 'models'
OUTPUT_ROOT = BASE_DIR / 'attack_data_tcomplex'
DATASETS = ['ICEWS05-15','ICEWS14','YAGO11k','WIKIDATA12k']
ICEWS_DATASETS = {'ICEWS05-15','ICEWS14'}
YEAR_DATASETS = {'YAGO11k','WIKIDATA12k'}
BATCH_SIZE = 4096
SEED = 42
random.seed(SEED); torch.manual_seed(SEED)

class TComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, num_times, embedding_dim):
        super().__init__()
        self.ent_re = nn.Embedding(num_entities, embedding_dim)
        self.ent_im = nn.Embedding(num_entities, embedding_dim)
        self.rel_re = nn.Embedding(2 * num_relations, embedding_dim)
        self.rel_im = nn.Embedding(2 * num_relations, embedding_dim)
        self.time_re = nn.Embedding(num_times, embedding_dim)
        self.time_im = nn.Embedding(num_times, embedding_dim)

    @staticmethod
    def complex_mul(a_re, a_im, b_re, b_im):
        return a_re*b_re-a_im*b_im, a_re*b_im+a_im*b_re

    def query_embedding(self, h, r, t):
        h_re, h_im = self.ent_re(h), self.ent_im(h)
        r_re, r_im = self.rel_re(r), self.rel_im(r)
        t_re, t_im = self.time_re(t), self.time_im(t)
        q_re, q_im = self.complex_mul(h_re,h_im,r_re,r_im)
        return self.complex_mul(q_re,q_im,t_re,t_im)

    def score(self, h, r, o, t):
        q_re, q_im = self.query_embedding(h,r,t)
        o_re, o_im = self.ent_re(o), self.ent_im(o)
        return torch.sum(q_re*o_re + q_im*o_im, dim=1)

def load_facts(path):
    facts=[]
    with open(path,'r',encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            parts=line.split()
            if len(parts)!=4: raise ValueError(f'{path} line {n}: expected 4 columns')
            facts.append(tuple(map(int,parts)))
    return facts

def load_time_mapping(dataset):
    p=PROCESSED_ROOT/dataset/'time2id.json'
    with open(p,'r',encoding='utf-8') as f: raw=json.load(f)
    time2id={str(k):int(v) for k,v in raw.items()}
    id2time={int(v):str(k) for k,v in raw.items()}
    return time2id,id2time

def get_neighbor_time_id(dataset, original_time_id, delta, time2id, id2time):
    original=id2time[int(original_time_id)]
    if dataset in ICEWS_DATASETS:
        d=datetime.strptime(original,'%Y-%m-%d')+timedelta(days=delta)
        value=d.strftime('%Y-%m-%d')
    elif dataset in YEAR_DATASETS:
        value=str(int(original)+delta)
    else:
        raise ValueError(dataset)
    return time2id.get(value)

def clean_state_dict(sd):
    out={}
    for k,v in sd.items():
        while k.startswith('_module.'): k=k[len('_module.'):]
        while k.startswith('module.'): k=k[len('module.'):]
        out[k]=v
    return out

def load_model(dataset, role, device):
    path=MODEL_ROOT/dataset/f'tcomplex_{role}.pt'
    if not path.exists(): raise FileNotFoundError(f'Missing model: {path}')
    ckpt=torch.load(path,map_location=device)
    if isinstance(ckpt,dict) and 'model_state_dict' in ckpt: sd=ckpt['model_state_dict']
    elif isinstance(ckpt,dict) and 'state_dict' in ckpt: sd=ckpt['state_dict']
    else: sd=ckpt
    sd=clean_state_dict(sd)
    ne=ckpt.get('num_entities',sd['ent_re.weight'].shape[0]) if isinstance(ckpt,dict) else sd['ent_re.weight'].shape[0]
    nr=ckpt.get('num_relations',sd['rel_re.weight'].shape[0]//2) if isinstance(ckpt,dict) else sd['rel_re.weight'].shape[0]//2
    nt=ckpt.get('num_times',sd['time_re.weight'].shape[0]) if isinstance(ckpt,dict) else sd['time_re.weight'].shape[0]
    dim=ckpt.get('embedding_dim',sd['ent_re.weight'].shape[1]) if isinstance(ckpt,dict) else sd['ent_re.weight'].shape[1]
    model=TComplEx(int(ne),int(nr),int(nt),int(dim))
    model.load_state_dict(sd,strict=True)
    model.to(device).eval()
    print('Checkpoint            :',path)
    return model

def score_queries(model, queries, device):
    scores=[]
    with torch.no_grad():
        for start in range(0,len(queries),BATCH_SIZE):
            x=torch.tensor(queries[start:start+BATCH_SIZE],dtype=torch.long,device=device)
            s=model.score(x[:,0],x[:,1],x[:,2],x[:,3])
            scores.extend(s.detach().cpu().tolist())
    return scores

def extract_features(dataset, model, facts, label, time2id, id2time, device):
    rows=[]; offsets=[-2,-1,0,1,2]
    for start in range(0,len(facts),BATCH_SIZE):
        batch=facts[start:start+BATCH_SIZE]
        queries=[]; flags=[]
        for h,r,o,t in batch:
            current=[]
            for delta in offsets:
                nt=get_neighbor_time_id(dataset,t,delta,time2id,id2time)
                if nt is None:
                    queries.append((h,r,o,t)); current.append(False)
                else:
                    queries.append((h,r,o,nt)); current.append(True)
            flags.append(current)
        scores=torch.tensor(score_queries(model,queries,device),dtype=torch.float32).reshape(len(batch),5)
        for i,(h,r,o,t) in enumerate(batch):
            vals=scores[i].tolist()
            for j in range(5):
                if not flags[i][j]: vals[j]=float('nan')
            rows.append({'h':h,'r':r,'o':o,'t':t,
                         'score_t_minus_2':vals[0],'score_t_minus_1':vals[1],
                         'score_t':vals[2],'score_t_plus_1':vals[3],'score_t_plus_2':vals[4],
                         'label':label})
    return rows

def balance_facts(members, nonmembers):
    n=min(len(members),len(nonmembers)); rng=random.Random(SEED)
    return rng.sample(members,n), rng.sample(nonmembers,n)

def build(dataset, role, member_file, nonmember_file, out_file, device):
    print('\n'+'='*78); print(f'{dataset} | {role.upper()} | TCOMPLEX'); print('='*78)
    ddir=PROCESSED_ROOT/dataset
    time2id,id2time=load_time_mapping(dataset)
    model=load_model(dataset,role,device)
    members=load_facts(ddir/member_file); nonmembers=load_facts(ddir/nonmember_file)
    print('Original members     :',len(members)); print('Original non-members :',len(nonmembers))
    members,nonmembers=balance_facts(members,nonmembers)
    print('Balanced             :',len(members),'+',len(nonmembers))
    print('Scoring members...'); rows=extract_features(dataset,model,members,1,time2id,id2time,device)
    print('Scoring non-members...'); rows+=extract_features(dataset,model,nonmembers,0,time2id,id2time,device)
    df=pd.DataFrame(rows).sample(frac=1,random_state=SEED).reset_index(drop=True)
    outdir=OUTPUT_ROOT/dataset; outdir.mkdir(parents=True,exist_ok=True)
    outpath=outdir/out_file; df.to_csv(outpath,index=False)
    cols=['score_t_minus_2','score_t_minus_1','score_t','score_t_plus_1','score_t_plus_2']
    print('Samples               :',len(df)); print('Temporal NaN rows     :',df[cols].isna().any(axis=1).sum())
    clean=df.dropna(subset=cols)
    if len(clean):
        diffs=[float((clean[c]-clean['score_t']).abs().max()) for c in ['score_t_minus_2','score_t_minus_1','score_t_plus_1','score_t_plus_2']]
        print('Max temporal diffs    :',diffs)
        if max(diffs)<1e-8: raise RuntimeError('Temporal scores collapsed; check model/scoring.')
    print('Saved                 :',outpath)

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('='*78); print('TCOMPLEX TEMPORAL ATTACK DATASET GENERATION'); print('='*78); print('Device:',device)
    ok=[]; bad=[]
    for d in DATASETS:
        try:
            build(d,'shadow','shadow_train.txt','shadow_test.txt','shadow_attack_dataset.csv',device)
            build(d,'victim','victim_train.txt','victim_test.txt','victim_attack_dataset.csv',device)
            ok.append(d)
        except Exception as e:
            print(f'\nERROR for {d}: {e!r}'); bad.append(d)
        finally:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    print('\nSuccessful:',ok); print('Failed:',bad)
