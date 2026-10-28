#!/usr/bin/env python3
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path('/home/harrouch/Paper2/Paper2')
PROCESSED_ROOT = BASE_DIR / 'processed'
MODEL_ROOT = BASE_DIR / 'models'
OUTPUT_ROOT = BASE_DIR / 'attack_data_tntcomplex'
DATASETS = ['ICEWS05-15','ICEWS14','YAGO11k','WIKIDATA12k']
SEED = 42
MAX_PER_CLASS = None
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_facts(path):
    facts=[]
    with open(path,'r',encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            p=line.split()
            if len(p)<4: raise ValueError(f'{path}:{i}: expected 4 columns')
            facts.append(tuple(map(int,p[:4])))
    if not facts: raise ValueError(f'No facts found in {path}')
    return facts

def load_time2id(dataset):
    with open(PROCESSED_ROOT/dataset/'time2id.json','r',encoding='utf-8') as f:
        raw=json.load(f)
    return {str(k):int(v) for k,v in raw.items()}

def invert_time2id(m): return {int(v):str(k) for k,v in m.items()}
def is_icews(dataset): return dataset.startswith('ICEWS')

def shift_real_time(value, delta, dataset):
    if is_icews(dataset):
        dt=pd.Timestamp(value)+pd.Timedelta(days=delta)
        return dt.strftime('%Y-%m-%d')
    return str(int(str(value))+delta)

def neighbor_time_id(tid, delta, dataset, id2time, time2id):
    if tid not in id2time: return None
    shifted=shift_real_time(id2time[tid],delta,dataset)
    return time2id.get(str(shifted))

def complex_mul(a_re,a_im,b_re,b_im):
    return a_re*b_re-a_im*b_im, a_re*b_im+a_im*b_re

class TNTComplEx(nn.Module):
    def __init__(self,num_entities,num_relations_with_reciprocals,num_times,embedding_dim):
        super().__init__()
        self.ent_re=nn.Embedding(num_entities,embedding_dim)
        self.ent_im=nn.Embedding(num_entities,embedding_dim)
        self.rel_time_re=nn.Embedding(num_relations_with_reciprocals,embedding_dim)
        self.rel_time_im=nn.Embedding(num_relations_with_reciprocals,embedding_dim)
        self.rel_static_re=nn.Embedding(num_relations_with_reciprocals,embedding_dim)
        self.rel_static_im=nn.Embedding(num_relations_with_reciprocals,embedding_dim)
        self.time_re=nn.Embedding(num_times,embedding_dim)
        self.time_im=nn.Embedding(num_times,embedding_dim)
    def score(self,h,r,o,t):
        h_re,h_im=self.ent_re(h),self.ent_im(h)
        rt_re,rt_im=self.rel_time_re(r),self.rel_time_im(r)
        rs_re,rs_im=self.rel_static_re(r),self.rel_static_im(r)
        time_re,time_im=self.time_re(t),self.time_im(t)
        dyn_re,dyn_im=complex_mul(rt_re,rt_im,time_re,time_im)
        rel_re,rel_im=dyn_re+rs_re,dyn_im+rs_im
        q_re,q_im=complex_mul(h_re,h_im,rel_re,rel_im)
        o_re,o_im=self.ent_re(o),self.ent_im(o)
        return torch.sum(q_re*o_re+q_im*o_im,dim=1)

def load_model(dataset,role,device):
    path=MODEL_ROOT/dataset/f'tntcomplex_{role}.pt'
    if not path.exists(): raise FileNotFoundError(path)
    ckpt=torch.load(path,map_location=device)
    model=TNTComplEx(ckpt['num_entities'],ckpt['num_relations_with_reciprocals'],ckpt['num_times'],ckpt['embedding_dim']).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    return model

@torch.no_grad()
def score_one(model,h,r,o,t,device):
    H=torch.tensor([h],dtype=torch.long,device=device)
    R=torch.tensor([r],dtype=torch.long,device=device)
    O=torch.tensor([o],dtype=torch.long,device=device)
    T=torch.tensor([t],dtype=torch.long,device=device)
    return float(model.score(H,R,O,T).item())

def sample_balanced(members,nonmembers,seed=SEED):
    rng=random.Random(seed)
    n=min(len(members),len(nonmembers))
    if MAX_PER_CLASS is not None: n=min(n,MAX_PER_CLASS)
    a=rng.sample(members,n); b=rng.sample(nonmembers,n)
    ex=[(x,1) for x in a]+[(x,0) for x in b]
    rng.shuffle(ex)
    return ex

def build_role(dataset,role,member_split,nonmember_split,device):
    print('\n'+'='*100); print(f'TNTComplEx STANDARD ATTACK DATA | {dataset} | {role.upper()}'); print('='*100)
    d=PROCESSED_ROOT/dataset
    members=read_facts(d/f'{member_split}.txt'); nonmembers=read_facts(d/f'{nonmember_split}.txt')
    ex=sample_balanced(members,nonmembers)
    print(f'Members available: {len(members)} | Non-members available: {len(nonmembers)} | Sampled/class: {len(ex)//2}')
    model=load_model(dataset,role,device)
    time2id=load_time2id(dataset); id2time=invert_time2id(time2id)
    rows=[]
    cols=[(-2,'score_t_minus_2'),(-1,'score_t_minus_1'),(0,'score_t'),(1,'score_t_plus_1'),(2,'score_t_plus_2')]
    for i,((h,r,o,t),label) in enumerate(ex,1):
        scores={}
        for delta,col in cols:
            cand=t if delta==0 else neighbor_time_id(t,delta,dataset,id2time,time2id)
            scores[col]=np.nan if cand is None else score_one(model,h,r,o,cand,device)
        rows.append({'h':h,'r':r,'o':o,'t':t,'label':label,**scores})
        if i%10000==0 or i==len(ex): print(f'Scored {i}/{len(ex)}')
    df=pd.DataFrame(rows)
    outdir=OUTPUT_ROOT/dataset; outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/f'{role}_attack_dataset.csv'; df.to_csv(out,index=False)
    full=df[[c for _,c in cols]].notna().all(axis=1)
    print(f'Rows saved: {len(df)} | Full-window rows: {int(full.sum())} ({100*full.mean():.2f}%)')
    print(f'Saved: {out}')
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__=='__main__':
    set_seed(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('='*100); print('BUILD TNTComplEx STANDARD MIA / T-MIA ATTACK DATASETS'); print('='*100); print(f'Device: {device}')
    if torch.cuda.is_available(): print('GPU   : '+torch.cuda.get_device_name(0))
    for ds in DATASETS:
        build_role(ds,'shadow','shadow_train','shadow_test',device)
        build_role(ds,'victim','victim_train','victim_test',device)
    print('\n'+'='*100); print('ALL TNTComplEx STANDARD ATTACK DATASETS BUILT'); print('='*100)
