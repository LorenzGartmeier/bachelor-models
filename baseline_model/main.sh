#!/bin/bash


python3 training/train_approximator.py
python3 training/create_selfdescriptions.py
python3 training/train_attributor.py
python3 evaluation/evaluation.py