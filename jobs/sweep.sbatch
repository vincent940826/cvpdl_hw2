#!/bin/bash
#SBATCH --account=ACD115240
#SBATCH --job-name=hw2-sweep
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u2951570/cvpdl_hw2/logs/%x-%j.out
#SBATCH --error=/home/u2951570/cvpdl_hw2/logs/%x-%j.err
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn006,25a-hgpn007,25a-hgpn008,25a-hgpn009

# Run several configs in one job (avoids the per-user submit limit).
# Usage: sbatch --export=NONE jobs/sweep.sbatch <list file>
# Each non-empty line of the list file: <method>|<splits>|<cfg JSON>|<out dir relative to starter/>
#   e.g. bn_ema|val test|{"momentum": 0.1}|../runs/sweep/bn_ema_m0.1
# Every run uses --diag. A failing run is reported and the sweep continues.

LIST=$(realpath "$1")
ROOT=/home/u2951570/cvpdl_hw2

module purge
module load cuda/12.6
module load miniconda3/26.1.1
eval "$(conda shell.bash hook)"
conda activate cvpdl

echo "host=$(hostname) list=$LIST"
nvidia-smi --query-gpu=name,clocks_throttle_reasons.hw_power_brake_slowdown --format=csv,noheader
python -c "import torch; torch.zeros(1, device='cuda'); print('CUDA ok:', torch.cuda.get_device_name())" \
  || { echo "CUDA preflight failed on $(hostname); resubmit with --exclude=$(hostname)"; exit 1; }

cd "$ROOT/starter"
while IFS='|' read -r METHOD SPLITS CFG OUT; do
  [ -z "$METHOD" ] && continue
  for SPLIT in $SPLITS; do
    echo "=== $METHOD $SPLIT $CFG -> $OUT"
    python run.py --method "$METHOD" --split "$SPLIT" --data ../data --ckpt ../source_resnet50_160.pth \
      --cfg "$CFG" --out "$OUT" --diag || echo "RUN FAILED: $METHOD $SPLIT $CFG"
  done
done < "$LIST"
echo "=== sweep done"
