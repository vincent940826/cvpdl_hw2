"""Run a TTA method on the val or test stream with the official harness.

  python run.py --method source --split val      # labeled: prints accuracy
  python run.py --method source --split test     # writes out/test_pred.csv for Kaggle
"""
import argparse
import json
import random

import numpy as np
import torch
import torchvision

from harness import run_stream
from methods import METHODS


def load_model(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = getattr(torchvision.models, ck["arch"])(num_classes=len(ck["wnids"]))
    model.load_state_dict(ck["state_dict"])
    return model.cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="source", choices=sorted(METHODS))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="source_resnet50_160.pth")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diag", action="store_true", help="also save per-batch diagnostics (analysis runs only)")
    ap.add_argument("--cfg", default="{}", help='method hyperparameters as JSON, e.g. \'{"alpha": 0.3}\'')
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    method = METHODS[args.method](load_model(args.ckpt), **json.loads(args.cfg))
    if args.diag:
        from diag import Recorder
        method = Recorder(method, load_model(args.ckpt))
    run_stream(method, args.data, args.split, args.out, extra_log=vars(args))
    if args.diag:
        method.save(f"{args.out}/{args.split}_diag.npz")


if __name__ == "__main__":
    main()
