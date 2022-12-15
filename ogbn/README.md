GARNET
===============================

GARNET on OGB datasets.

Example Usage
-----

* GARNET-GCN on arxiv (clean): \
`python main.py --dataset arxiv`

* GARNET-GCN on arxiv (GR-BCD w/ 25% perturbation ratio): \
`python main.py --dataset arxiv --ptb_rate 0.25 --perturbed`

* GARNET-GCN on arxiv (GR-BCD w/ 50% perturbation ratio): \
`python main.py --dataset arxiv --ptb_rate 0.5 --perturbed`

* GARNET-GCN on products (clean): \
`python main.py --dataset products`

* GARNET-GCN on products (GR-BCD w/ 25% perturbation ratio): \
`python main.py --dataset products --ptb_rate 0.25 --perturbed`

* GARNET-GCN on products (GR-BCD w/ 50% perturbation ratio): \
`python main.py --dataset products --ptb_rate 0.5 --perturbed`
