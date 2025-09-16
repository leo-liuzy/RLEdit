#!/bin/bash
#SBATCH -J rledit       # Job name
#SBATCH -o slurm-outputs/%x.o%j       # Name of stdout output file
#SBATCH -e slurm-outputs/%x.e%j       # Name of stderr output file
#SBATCH -p gh          # Queue (partition) name
#SBATCH -N 1              # Total # of nodes
##SBATCH --ntasks-per-node=1
#SBATCH -t 20:00:00        # Run time (hh:mm:ss)
#SBATCH -A CCR25005       # Allocation name (req'd if you have more than 1)

# CUDA_VISIBLE_DEVICES=0 python main.py dataset=zsre model=llama-3-instruct editor=rledit num_seq=400
# CUDA_VISIBLE_DEVICES=0 python main.py dataset=cre model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-4-15 editor=rledit num_seq=100 editor.n_epochs=1 dataset.n_edits=20 dataset.batch_size=4  # editor.cache_dir=cache2

# CUDA_VISIBLE_DEVICES=0 python main.py dataset=cre model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-4-15 editor=rledit num_seq=100 editor.n_epochs=1 dataset.n_edits=10 dataset.batch_size=5

# CUDA_VISIBLE_DEVICES=0 python main.py dataset=cre model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-4-15 editor=rledit num_seq=100 editor.n_epochs=1 dataset.n_edits=10 dataset.batch_size=5 editor.back_depth=1 editor.cache_dir=cache2

# CUDA_VISIBLE_DEVICES=0 python main.py dataset=cre model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-4-15 editor=rledit num_seq=20 editor.n_epochs=10 dataset.n_edits=1 dataset.batch_size=1 editor.back_depth=10

# CUDA_VISIBLE_DEVICES=0 python main.py dataset=zsre model=llama-3.2-instruct editor=rledit num_seq=200 editor.n_epochs=20

# CUDA_VISIBLE_DEVICES=0 python main.py dataset=zsre model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-4-15 editor=rledit num_seq=100 editor.n_epochs=1 dataset.n_edits=10 dataset.batch_size=5
export CUDA_VISIBLE_DEVICES=7
python eval_propmend.py dataset=cre model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-4-15 editor=rledit num_seq=8 editor.n_epochs=10 dataset.n_edits=5 editor.load_checkpoint=True