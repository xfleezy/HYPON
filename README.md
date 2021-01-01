# HYPON
Pytorch implementation for HYPON: Embedding Biomedical Ontology using Hyperbolic Ontology Neural network.

## Abstract
Constructing high-quality biomedical ontologies is one of the first  steps  to  study  new  concepts,  such  as  emerging  infectious diseases. Manually curated ontologies are often noisy, especially  for  new  knowledge  that  requires  domain  expertise. In this paper, we proposed a novel ontology embedding approach  HYPON  to  automate  this  process.  In  contrast  to conventional  approaches,  we  propose  to  embed  biomedical ontology in the hyperbolic space to better model the hierarchical structure. Importantly, our method is able to consider both  graph  structure  and  the  varied-size  set  of  concept  instances, which are largely overlooked by existing methods. We demonstrated substantial improvement in comparison to thirteen comparison approaches on eleven biomedical ontologies, including two recently curated COVID-19 ontologies.

## Model flowchart

HYPON takes an ontology graph as input. Each node on the graph is associated with avaried-size set of concept instances (e.g., patients). HYPON first split each node into subnodes according to the number ofconcept instances. It then performed a bidirectional message passing to aggregate information from parent nodes and childnodes separately. These subnodes are then merged together according to their embeddings in the Poincar ́e ball. Finally, HYPONpredicts new links in the hyperbolic space using merged embeddings.

## Requirements
conda env create -f environment.yml

## Experiments
1. _HYPON_ * CL (test AUROC = 0.88)
``` python train.py --task lp --dataset cl --model HYPON --lr 0.008 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold PoincareBall --log-freq 5 --cuda 0```
  _HGCN_ * CL (test AUROC = 0.77)
```python train.py --task lp --dataset cl --model HGCN --lr 0.008 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold PoincareBall --log-freq 5 --cuda 0```
