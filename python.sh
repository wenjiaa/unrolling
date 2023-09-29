#!/bin/bash

#SBATCH --job-name=unrolling

#SBATCH --partition=gpu # partition souhaitée

#SBATCH --gres=gpu:1

#SBATCH --time=2-00:00:00 # job lancé pour 2 jours max

#SBATCH --mem=50GB # mémoire demandée

#SBATCH --output=/feynman/work/dedip/lilas/wf274757/unrolling/python.log 

module load anaconda 

source activate /feynman/home/dedip/lilas/wf274757/work/pyenv3

python /feynman/work/dedip/lilas/wf274757/unrolling/codes/training.py --Model='GLPALM' --Dataset='mixture_sigma0.1' --var_Scaling=1. --NLayers=1 --NEpochs=1000 --Version='CNN'
# python /feynman/work/dedip/lilas/wf274757/unrolling/codes/training.py --Model='LPALM' --Dataset='mixture_sigma0.1' --var_Scaling=0. --NLayers=5 --update_A='LS' --W_shape='matrix'

# # No-updating => GLFBS/LFBS 
# python /feynman/work/dedip/lilas/wf274757/unrolling/codes/training.py --Model='GLPALM' --Dataset='noisy_Cs137' --NLayers=8 --NEpochs=1000 --Version='CNN' --update_A='No-updating' 