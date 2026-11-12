"""Official online TTA harness for CVPDL HW2. Do not modify this file.

The stream is served strictly in order, one batch at a time. Batch k+1 is handed to your method
only after it has returned predictions for batch k, and predictions cannot be changed afterwards.
Peak GPU memory of this process is sampled from nvidia-smi during the whole run (limit: 10 GB).
Wall-clock time of the run is logged too (limit: 90 seconds for the test stream when the TAs rerun it on
an RTX 3090).
"""
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BATCH_SIZE = 64
VRAM_LIMIT_MB = 10 * 1024


class _Images(Dataset):
    def __init__(self, img_dir, ids):
        self.img_dir, self.ids = Path(img_dir), ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x = np.asarray(Image.open(self.img_dir / f"{self.ids[i]}.png").convert("RGB"))
        return torch.from_numpy(x.copy()).permute(2, 0, 1)


class VRAMMonitor(threading.Thread):
    """Polls nvidia-smi for this process' GPU memory (MiB) and keeps the peak."""

    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval, self.peak_mb, self._halt = interval, 0.0, threading.Event()

    def sample(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=5).stdout
        except Exception:
            return
        for line in out.strip().splitlines():
            pid, mb = [v.strip() for v in line.split(",")]
            if pid.isdigit() and int(pid) == os.getpid():
                self.peak_mb = max(self.peak_mb, float(mb))

    def run(self):
        while not self._halt.is_set():
            self.sample(); time.sleep(self.interval)

    def stop(self):
        self._halt.set(); self.join(); self.sample()


def run_stream(method, data_root, split, out_dir, num_workers=4, extra_log=None):
    """Feed the `split` stream ("val" or "test") to `method` and write predictions + run log.

    method(x) receives a normalized float CUDA tensor (B, 3, H, W) and must return a LongTensor/array
    of B labels in {0, ..., 19}.
    """
    data_root, out_dir = Path(data_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(data_root / split / "stream.csv")["id"].tolist()
    loader = DataLoader(_Images(data_root / split / "images", ids), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers)
    mon = VRAMMonitor(); mon.start()
    preds, t0 = [], time.time()
    for k, xb in enumerate(loader):
        x = ((xb.float() / 255 - MEAN) / STD).cuda(non_blocking=True)
        p = method(x)
        p = (p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p)).astype(np.int64).reshape(-1)
        assert len(p) == len(xb), f"batch {k}: got {len(p)} predictions for {len(xb)} images"
        assert ((p >= 0) & (p <= 19)).all(), f"batch {k}: labels must be in 0..19"
        preds.append(p)
        if (k + 1) % 20 == 0:
            print(f"[{split}] {(k + 1) * BATCH_SIZE}/{len(ids)}  peak VRAM {mon.peak_mb:.0f} MiB", flush=True)
    torch.cuda.synchronize(); seconds = time.time() - t0; mon.stop()
    max_reserved = torch.cuda.max_memory_reserved() / 2 ** 20
    preds = np.concatenate(preds)
    sub = pd.DataFrame(dict(id=ids, label=preds))
    sub.to_csv(out_dir / f"{split}_pred.csv", index=False)
    log = dict(split=split, n=len(ids), seconds=round(seconds, 1),
               peak_vram_mib=mon.peak_mb, vram_ok=bool(0 < mon.peak_mb <= VRAM_LIMIT_MB),
               torch_max_reserved_mib=round(max_reserved, 1),
               gpu=torch.cuda.get_device_name(), torch=torch.__version__, python=platform.python_version(),
               pred_sha256=hashlib.sha256(sub.to_csv(index=False).encode()).hexdigest(), config=extra_log or {})
    json.dump(log, open(out_dir / f"{split}_run_log.json", "w"), indent=1)
    labels = data_root / split / "labels.csv"
    if labels.exists():
        y = pd.read_csv(labels).set_index("id").loc[ids, "label"].to_numpy()
        log["acc"] = float((preds == y).mean())
        json.dump(log, open(out_dir / f"{split}_run_log.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in log.items() if k not in ("pred_sha256",)}, indent=1))
    if not log["vram_ok"]:
        print(f"WARNING: peak VRAM {mon.peak_mb:.0f} MiB exceeds the 10 GB limit (or could not be read)")
    return log
