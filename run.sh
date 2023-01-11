#!/bin/bash

gnn='gprgnn'
dset='citeseer'

# Nettack (clean)
python main.py --device 0 --backbone $gnn --dataset $dset --attack nettack --ptb_rate 1.0;
# Nettack (adversarial)
python main.py --device 0 --backbone $gnn --dataset $dset --attack nettack --ptb_rate 5.0 --perturbed;

# Metattack (clean)
python main.py --device 0 --backbone $gnn --dataset $dset --attack meta --ptb_rate 0.1;
# Metattack (adversarial)
python main.py --device 0 --backbone $gnn --dataset $dset --attack meta --ptb_rate 0.2 --perturbed;
