# utils/constants.py
import os

# Directory for saving checkpoints
checkpoints_dir = './checkpoints/autos_cifar10'

# Directory for datasets
datasets_dir = './data/datasets'

# Ensure these directories exist
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(datasets_dir, exist_ok=True)

