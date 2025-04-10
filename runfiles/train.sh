#!/bin/bash
#SBATCH --partition=gshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --mem=24gb
#SBATCH --output=logs/sbatch/train_%j.log # Standard output log
#SBATCH --error=logs/sbatch/train_%j.log  # Standard error log
source venv/bin/activate

### LGATr ###
python run.py -cp config -cn lgatr-naive run.name=lgatr-test

### MLP ###
python run.py -cp config -cn mlp-lorentz run.name=mlp-lorentz-test