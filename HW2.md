# CVPDL HW2: Test-Time Adaptation

## Files

```
hw2/
├── HW2.md                       this file
├── source_resnet50_160.pth      source model (ResNet-50, 160x160 input, 20 classes)
├── starter/
│   ├── harness.py               official online harness (do not modify)
│   ├── run.py                   entry point
│   ├── methods.py               put your TTA method here (`source` = no adaptation)
│   └── classes.txt              class index and name
└── data/
    ├── val/
    │   ├── images/              2,112 images (160x160 PNG)
    │   ├── stream.csv           stream order
    │   └── labels.csv           labels (id,label)
    ├── test/
    │   ├── images/              12,288 images (160x160 PNG)
    │   └── stream.csv           stream order
    └── sample_submission.csv
```

## Quick start

```bash
cd starter
python run.py --method source --split val  --data ../data --ckpt ../source_resnet50_160.pth --out ../out   # prints val accuracy
python run.py --method source --split test --data ../data --ckpt ../source_resnet50_160.pth --out ../out   # writes ../out/test_pred.csv
```

Upload `test_pred.csv` to Kaggle.

## Submission format

- CSV with header `id,label`, 12,288 rows (one per test image), `label` is an integer in 0–19.
- See `data/sample_submission.csv`.
- Metric: accuracy.
- The public leaderboard is computed on 50% of the test data. Your final ranking uses the other 50% (private leaderboard).

## Kaggle rules

- **Deadline: 2026/11/20 23:59:59 (UTC+8)**
- 10 submissions per day.
- Remember to rename your **Team name** as your student number (e.g. R15922001).
- If your name is not your student ID, your leaderboard score will be 0.
- Do NOT use multiple accounts to submit results. Any violation will result in a score of 0.
- Start as early as possible! The pending might take time (~3 min).
