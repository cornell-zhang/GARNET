GARNET
===============================

GARNET is a scalable reduced-rank topology learning method for constructing GNN models robust to adversarial attacks on homo/heterophilic graphs. More details are available in our paper: https://openreview.net/forum?id=kvwWjYQtmw

![Overview of the GARNET framework](/GARNET.png)

Citation
------------
If you use GARNET in your research, please cite our work published at LoG'22.

```
@inproceedings{
deng2022garnet,
title={{GARNET}: Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks},
author={Chenhui Deng and Xiuyu Li and Zhuo Feng and Zhiru Zhang},
booktitle={Learning on Graphs Conference},
year={2022},
url={https://openreview.net/forum?id=kvwWjYQtmw}
}
```

Requirements
------------
* python 3.8 (we suggest [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) to manage package dependencies.)
* pytorch 1.11 (required to install torch_geometric using conda)
* torch_geometric
* opt_einsum
* deeprobust
* ogb
* pyyaml
* gdown

Installation
------------
* Follow the steps below to install all required packages. In step 3, you need to install pytorch with proper cuda version on your platform (we are using cuda 11.3).
```
1. conda create -n garnet python=3.8
2. conda activate garnet
3. conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
4. conda install pyg -c pyg
5. pip install -r requirements.txt
```

Available Dataset
-------
* Cora
* Citeseer (newly added on 1/10/2023)
* Pubmed
* Chameleon
* Squirrel
* ogbn-arxiv
* ogbn-products

Example Usage
-----

**Note**: We only show how to run GARNET-GCN on Cora dataset below. For other settings, you only need to change the names of dataset and backbone model.

* GARNET-GCN on clean Cora graph under Nettack test nodes: \
`python main.py --device 0 --backbone gcn --dataset cora --attack nettack --ptb_rate 1.0`

* GARNET-GCN on adversarial Cora graph under Nettack with 5 perturbation per target node: \
`python main.py --device 0 --backbone gcn --dataset cora --attack nettack --ptb_rate 5.0 --perturbed`

* GARNET-GCN on clean Cora graph under Metattack test nodes: \
`python main.py --device 0 --backbone gcn --dataset cora --attack meta --ptb_rate 0.1`

* GARNET-GCN on adversarial Cora graph under Metattack with 20% perturbation rate: \
`python main.py --device 0 --backbone gcn --dataset cora --attack meta --ptb_rate 0.2 --perturbed`

Experimental Results
-------

**Note**: We further tune some hyperparameters in GARNET and achieve even better results than what we report in our paper. The improved results on Cora dataset are shown in the table below. Thus, we recommend users to use the new hyperparameter setting of GARNET in their experiments (available in `configs/`).

| Method        | Accuracy reported in our paper      | Latest accuracy (12/15/2022)  |
| :-----------: |:-------------:| :-------:|
| GARNET-GCN-Net-Clean      |  81.08 ± 2.05   | 83.25 ± 1.51 (2.17%↑)  |
| GARNET-GCN-Net-Adv        |  67.04 ± 2.05   | 76.39 ± 1.16 (9.35%↑)  |
| GARNET-GCN-Meta-Clean     |  79.64 ± 0.75   | 81.90 ± 0.34 (2.26%↑)  |
| GARNET-GCN-Meta-Adv       |  73.89 ± 0.91   | 76.23 ± 0.87 (2.34%↑)  |

Experiments on Large Graphs
-------

```
1. cd ogbn/
2. see README.md for instructions
```
