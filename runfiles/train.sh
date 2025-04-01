#!/bin/bash
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --mem=24gb
#SBATCH --output=logs/sbatch/train_%j.log # Standard output log
#SBATCH --error=logs/sbatch/train_%j.log  # Standard error log
source venv/bin/activate

### LGATr runs ###
# minimal
# python run.py -cp config -cn lgatr-naive run.name=lgatr-naive-hs dataset.process=gg_5g model.nepochs=1 dataset.trn_tst_val=[0.01,0.01,0.01]
# full
python run.py -cp config -cn lgatr-naive run.name=lgatr-naive-hs-ufcs_true dataset.process=gg_5g


### MLP runs ###
# minimal
# python run.py -cp config -cn mlp-lorentz run.name=mlp-lorentz-hs-test dataset.process=gg_5g model.nepochs=1 dataset.trn_tst_val=[0.01,0.01,0.01]

# full
# python run.py -cp config -cn mlp-lorentz run.name=mlp-lorentz-hs dataset.process=gg_5g