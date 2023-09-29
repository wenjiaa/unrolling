#!/bin/bash

#SBATCH --job-name=notebook

#SBATCH --partition=gpu_interactive

#SBATCH --gres=shard:1

#SBATCH --time=08:00:00 

#SBATCH --mem=50GB 

#SBATCH --output=/feynman/work/dedip/lilas/wf274757/unrolling/jupyterlab.log # log file

module load anaconda 

source activate /feynman/home/dedip/lilas/wf274757/work/pyenv3

cd /feynman/work/dedip/lilas/wf274757/unrolling/notebooks
# jupyter notebook --no-browser --port=8001 #--NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0'
jupyter lab --ip="$(hostname -I|sed -e 's/.*\(10\.2\.10[45]\.[[:digit:]]*\).*/\1/')" --port=8001