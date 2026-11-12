#!/bin/bash
#SBATCH --account=ACD115240
#SBATCH --job-name=hw2-run
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/u2951570/cvpdl_hw2/logs/%x-%j.out
#SBATCH --error=/home/u2951570/cvpdl_hw2/logs/%x-%j.err
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009

# Run one TTA method on the val and/or test stream with the official harness.
# Usage: sbatch --export=NONE jobs/run.sbatch <method> ["val test"] [extra run.py args]
#   e.g. sbatch --export=NONE jobs/run.sbatch tent "val test" --diag --out ../runs/diag/tent
# Outputs: runs/<method>/{val,test}_pred.csv and {val,test}_run_log.json

set -e
METHOD=${1:-source}
SPLITS=${2:-"val test"}
shift $(( $# < 2 ? $# : 2 ))   # the rest goes to run.py; a later --out overrides the default
EXTRA=("$@")
ROOT=/home/u2951570/cvpdl_hw2

module purge
module load cuda/12.6
module load miniconda3/26.1.1
eval "$(conda shell.bash hook)"
conda activate cvpdl

echo "host=$(hostname) method=$METHOD splits=$SPLITS"
nvidia-smi --query-gpu=name,clocks_throttle_reasons.hw_power_brake_slowdown --format=csv,noheader
python -c "import torch; torch.zeros(1, device='cuda'); print('CUDA ok:', torch.cuda.get_device_name())" \
  || { echo "CUDA preflight failed on $(hostname); resubmit with --exclude=$(hostname)"; exit 1; }

cd "$ROOT/starter"
for SPLIT in $SPLITS; do
  python run.py --method "$METHOD" --split "$SPLIT" --data ../data \
    --ckpt ../source_resnet50_160.pth --out "../runs/$METHOD" "${EXTRA[@]}"
done
