import os

out = []
with open('entity2id.txt','r') as f:
    lines=f.readlines()
    for l in lines:
        ent, id = l.strip().split('\t')
        out.append("{}\t{}\n".format(id, ent))

with open('readable_entities.txt','w') as f:
    for l in out:
        f.write(l)

out = []
with open('relation2id.txt','r') as f:
    lines=f.readlines()
    for l in lines:
        ent, id = l.strip().split('\t')
        out.append("{}\t{}\n".format(id, ent))

with open('readable_relations.txt','w') as f:
    for l in out:
        f.write(l)
