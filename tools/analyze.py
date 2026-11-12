"""Offline analysis of TTA runs: accuracy breakdowns on val and label-free health metrics on val/test.

Usage (CPU only, a few seconds, fine on the login node):
  python tools/analyze.py runs/diag/source runs/diag/bn_adapt runs/diag/tent [--seg 11] [--fig runs/analysis]

Each run dir holds {val,test}_pred.csv and {val,test}_run_log.json written by run.py. If
{val,test}_diag.npz exists (run.py --diag), it adds entropy/confidence, BN-param drift and the
feature-shift signal. Reads data/val/labels.csv for analysis only; nothing here feeds back into a method.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
B, C = 64, 20
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # categorical slots, fixed order
GRAY = "#8c8b86"
CLASSES = [line.split()[2] for line in open(ROOT / "starter" / "classes.txt")]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "legend.frameon": False})


def load_run(d):
    d = Path(d)
    r = dict(name=d.name, ids={}, pred={}, log={}, diag={})
    for s in ("val", "test"):
        if (d / f"{s}_pred.csv").exists():
            df = pd.read_csv(d / f"{s}_pred.csv")
            r["ids"][s], r["pred"][s] = df["id"].tolist(), df["label"].to_numpy()
            r["log"][s] = json.load(open(d / f"{s}_run_log.json"))
        if (d / f"{s}_diag.npz").exists():
            r["diag"][s] = dict(np.load(d / f"{s}_diag.npz"))
    return r


def segments(n, length):
    return [(i, min(i + length, n)) for i in range(0, n, length)]


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)


def title(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}")


def show(df, fmt="{:.3f}"):
    print(df.to_string(float_format=lambda v: fmt.format(v)))


# ---------------------------------------------------------------- resources
def resources(runs):
    title("Resources (diag runs include the recorder's extra forward pass, so their seconds are inflated)")
    rows = [dict(run=r["name"], split=s, seconds=g["seconds"], peak_vram_mib=g["peak_vram_mib"],
                 vram_ok=g["vram_ok"], torch_reserved_mib=g["torch_max_reserved_mib"], gpu=g["gpu"],
                 diag=g["config"].get("diag", False)) for r in runs for s, g in r["log"].items()]
    show(pd.DataFrame(rows).set_index(["run", "split"]), "{:.1f}")


# ---------------------------------------------------------------- val, labeled
def val_section(runs, y, seg, bounds=None):
    nb = len(y) // B
    yb = y.reshape(nb, B)
    present = [set(b.tolist()) for b in yb]
    ncls = np.array([len(s) for s in present])
    segs = list(zip([0] + bounds, bounds + [nb])) if bounds else segments(nb, seg)
    seg_names = [f"b{a}-{b - 1}" for a, b in segs]
    vr = [r for r in runs if "val" in r["pred"]]
    acc = {r["name"]: (r["pred"]["val"] == y).reshape(nb, B).mean(1) for r in vr}

    title(f"Val accuracy: overall and per domain (starts {[a for a, _ in segs]}; from --bounds or every --seg batches)")
    show(pd.DataFrame([{"run": n, "overall": a.mean(), **{sn: a[s0:s1].mean() for sn, (s0, s1) in zip(seg_names, segs)}}
                       for n, a in acc.items()]).set_index("run"))

    title("Val per-batch accuracy (n_cls = number of true classes in the batch)")
    show(pd.DataFrame({"n_cls": ncls, **acc}).rename_axis("batch"), "{:.2f}")

    title("Correlation over batches: accuracy vs n_cls (and accuracy gain over source vs n_cls)")
    src = next((r["name"] for r in vr if r["log"]["val"]["config"]["method"] == "source"), None)
    show(pd.DataFrame([{"run": n, "corr(acc, n_cls)": corr(a, ncls),
                        "corr(acc - source, n_cls)": corr(a - acc[src], ncls) if src and n != src else np.nan}
                       for n, a in acc.items()]).set_index("run"))

    title("Val accuracy by how often the image's class appears in its own batch")
    cnt = np.array([[np.sum(b == v) for v in b] for b in yb])
    bins = [(1, 2), (3, 5), (6, 9), (10, B)]
    rows = []
    for sn, (s0, s1) in [("all", (0, nb))] + list(zip(seg_names, segs)):
        for lo, hi in bins:
            m = np.zeros((nb, B), bool)
            m[s0:s1] = (cnt[s0:s1] >= lo) & (cnt[s0:s1] <= hi)
            rows.append({"batches": sn, "class count in batch": f"{lo}-{hi}", "n_img": int(m.sum()),
                         **{r["name"]: (r["pred"]["val"].reshape(nb, B) == yb)[m].mean() if m.any() else np.nan
                            for r in vr}})
    show(pd.DataFrame(rows).set_index(["batches", "class count in batch"]))

    title("Share of wrong predictions that name a class absent from the batch (chance = absent classes / 19)")
    chance_b = np.broadcast_to(((C - ncls) / (C - 1))[:, None], (nb, B))
    rows = []
    for r in vr:
        p = r["pred"]["val"].reshape(nb, B)
        wrong = p != yb
        absent = np.array([[v not in present[k] for v in p[k]] for k in range(nb)])
        row = {"run": r["name"], "n_wrong": int(wrong.sum())}
        for sn, (s0, s1) in [("all", (0, nb))] + list(zip(seg_names, segs)):
            w = wrong.copy(); w[:s0] = False; w[s1:] = False
            row[sn] = absent[w].mean() if w.any() else np.nan
            row[f"{sn} chance"] = chance_b[w].mean() if w.any() else np.nan
        rows.append(row)
    show(pd.DataFrame(rows).set_index("run"))

    title("Val per-class accuracy")
    show(pd.DataFrame({"class": CLASSES, "n": np.bincount(y, minlength=C),
                       **{r["name"]: [(r["pred"]["val"][y == c] == c).mean() for c in range(C)] for r in vr}}))

    title("Top confusions on val (true -> predicted, count)")
    for r in vr:
        p = r["pred"]["val"]
        pairs = pd.Series(list(zip(y[p != y], p[p != y]))).value_counts().head(6)
        print(f"{r['name']:>10}: " + "; ".join(f"{CLASSES[t]}->{CLASSES[q]} {n}" for (t, q), n in pairs.items()))

    title("Label structure of the val stream (method-independent)")
    jac = lambda a, b: len(a & b) / len(a | b)  # noqa: E731
    adj = [jac(present[k], present[k + 1]) for k in range(nb - 1)]
    far = [jac(present[i], present[j]) for i in range(nb) for j in range(i + 2, nb)]
    print(f"classes per batch: mean {ncls.mean():.1f}, min {ncls.min()}, max {ncls.max()}")
    print(f"Jaccard of class sets: consecutive batches {np.mean(adj):.3f} vs non-adjacent pairs {np.mean(far):.3f}")
    for k in (1, 2, 4, 8, 16):
        cov = [len(set().union(*present[i:i + k])) for i in range(nb - k + 1)]
        print(f"distinct classes in {k:>2} consecutive batches: mean {np.mean(cov):.1f}, min {min(cov)}")
    return acc, ncls, present, segs


# ---------------------------------------------------------------- label-free
def label_free(runs, split, acc=None, true_ncls=None, chunks=None):
    rs = [r for r in runs if split in r["pred"]]
    if not rs:
        return
    n = len(rs[0]["pred"][split])
    nb = n // B
    src = next((r for r in rs if r["log"][split]["config"]["method"] == "source"), None)
    title(f"Label-free metrics on {split} ({n} images)")
    rows = []
    for r in rs:
        p = r["pred"][split]
        h = np.bincount(p, minlength=C)
        d = r["diag"].get(split)
        row = {"run": r["name"], "max class": CLASSES[h.argmax()], "max/expected": h.max() / (n / C),
               "min/expected": h.min() / (n / C),
               "distinct/batch": np.mean([len(set(b)) for b in p.reshape(nb, B)]),
               "agree w/ source": (p == src["pred"][split]).mean() if src is not None else np.nan}
        if d is not None:
            ent, conf = entropy(d["probs"]), d["probs"].max(1)
            row.update({"entropy": ent.mean(), "confidence": conf.mean(), "drift end (x1e3)": 1e3 * d["drift"][-1]})
            if acc is not None:
                row["corr(conf, acc) per batch"] = corr(conf.reshape(nb, B).mean(1), acc[r["name"]])
        rows.append(row)
    show(pd.DataFrame(rows).set_index("run"))
    if true_ncls is not None:
        print(f"(true distinct classes per val batch: mean {true_ncls.mean():.1f})")

    chunks = chunks or segments(nb, nb // 4)
    names = [f"b{a}-{b - 1}" for a, b in chunks]
    title(f"{split}: mean entropy / max-class share / agreement with source, per chunk of the stream")
    rows = []
    for r in rs:
        p = r["pred"][split].reshape(nb, B)
        d = r["diag"].get(split)
        ent = entropy(d["probs"]).reshape(nb, B) if d is not None else None
        for metric in ("entropy", "max-class share", "agree w/ source"):
            row = {"run": r["name"], "metric": metric}
            for cn, (a, b) in zip(names, chunks):
                if metric == "entropy":
                    row[cn] = ent[a:b].mean() if ent is not None else np.nan
                elif metric == "max-class share":
                    row[cn] = np.bincount(p[a:b].ravel(), minlength=C).max() / p[a:b].size
                else:
                    row[cn] = (p[a:b] == src["pred"][split].reshape(nb, B)[a:b]).mean() if src is not None else np.nan
            rows.append(row)
    show(pd.DataFrame(rows).set_index(["run", "metric"]))


# ---------------------------------------------------------------- feature shift
def shift_curves(runs, split):
    r = next((r for r in runs if split in r["diag"]), None)
    if r is None:
        return None
    d = r["diag"][split]
    L = d["shift_src"].shape[1]
    groups = {"shallow": slice(0, L // 3), "deep": slice(L - L // 3, L)}
    return {f"{g} vs {ref}": d[key][:, sl].mean(1)  # row 0 of shift_prev is NaN by design
            for ref, key in (("source", "shift_src"), ("prev", "shift_prev")) for g, sl in groups.items()}


def shift_section(runs, split, acc=None, present=None, k=3.0):
    sc = shift_curves(runs, split)
    if sc is None:
        return None
    title(f"Feature shift on {split} (frozen source model; symmetric KL of BN-input stats, mean over layer group)")
    s = sc["shallow vs prev"]
    thr = k * np.nanmedian(s)
    jumps = [int(i) for i in np.where(s > thr)[0]]
    print(f"batches whose shallow-layer shift vs previous batch > {k:g} x median ({thr:.3f}): {jumps}")
    for i in jumps:
        print(f"  batch {i:>3}: shallow vs prev {s[i]:.3f}, deep vs prev {sc['deep vs prev'][i]:.3f}, "
              f"shallow vs source {sc['shallow vs source'][i]:.3f}")
    print("shallow vs source per chunk: " + ", ".join(
        f"b{a}-{b - 1} {sc['shallow vs source'][a:b].mean():.3f}" for a, b in segments(len(s), max(len(s) // 8, 1))))
    if acc is not None and present is not None:
        nb = len(present)
        change = np.array([np.nan] + [1 - len(present[i] & present[i - 1]) / len(present[i] | present[i - 1])
                                      for i in range(1, nb)])
        keep = np.ones(nb, bool); keep[jumps] = False
        src = next((n for n in acc if n == "source"), None)
        print("within-domain batches only (jump batches excluded):")
        print(f"  corr(label-set change vs prev batch, shallow shift vs prev) = {corr(change[keep], sc['shallow vs prev'][keep]):.2f}")
        print(f"  corr(label-set change vs prev batch, deep shift vs prev)    = {corr(change[keep], sc['deep vs prev'][keep]):.2f}")
        if src:
            print(f"  corr(source per-batch acc, shallow shift vs source) = {corr(acc[src], sc['shallow vs source']):.2f}")
    return jumps


# ---------------------------------------------------------------- figures
def figures(runs, y, acc, ncls, segs, out):
    out.mkdir(parents=True, exist_ok=True)
    color = {r["name"]: COLORS[i % len(COLORS)] for i, r in enumerate(runs)}
    nb = len(ncls)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    for n, a in acc.items():
        a1.plot(range(nb), a, marker="o", ms=4, color=color[n], label=n)
    for s0, _ in segs[1:]:
        for ax in (a1, a2):
            ax.axvline(s0 - 0.5, color=GRAY, ls="--", lw=1)
    a1.set_ylabel("accuracy"); a1.set_ylim(0, 1); a1.legend(loc="lower left", ncol=len(acc))
    a1.set_title("Val accuracy per batch (dashed: domain boundaries used for the breakdowns)", loc="left")
    a2.bar(range(nb), ncls, color=GRAY, width=0.7)
    a2.set_ylabel("true classes"); a2.set_xlabel("batch")
    fig.tight_layout(); fig.savefig(out / "val_per_batch.png", dpi=130); plt.close(fig)

    curves = {s: shift_curves(runs, s) for s in ("val", "test")}
    if any(v is not None for v in curves.values()):
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
        for i, s in enumerate(("val", "test")):
            for j, ref in enumerate(("source", "prev")):
                ax = axes[i, j]
                if curves[s] is None:
                    ax.set_visible(False); continue
                for k, g in enumerate(("shallow", "deep")):
                    ax.plot(curves[s][f"{g} vs {ref}"], color=COLORS[k], lw=1.6, label=f"{g} layers")
                ax.set_yscale("log"); ax.set_xlabel("batch")
                ax.set_title(f"{s}: batch stats vs {'source stats' if ref == 'source' else 'previous batch'}", loc="left")
                ax.legend(loc="upper right")
        fig.tight_layout(); fig.savefig(out / "feature_shift.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    for i, s in enumerate(("val", "test")):
        rs = [r for r in runs if s in r["pred"]]
        n_b = len(rs[0]["pred"][s]) // B if rs else 0
        for r in rs:
            if s in r["diag"]:
                axes[i, 0].plot(entropy(r["diag"][s]["probs"]).reshape(n_b, B).mean(1), color=color[r["name"]],
                                lw=1.6, label=r["name"])
            axes[i, 1].plot([len(set(b)) for b in r["pred"][s].reshape(n_b, B)], color=color[r["name"]], lw=1.6,
                            label=r["name"])
        if s == "val":
            axes[i, 1].plot(ncls, color=GRAY, ls="--", lw=1.4, label="true")
        axes[i, 0].set_title(f"{s}: mean prediction entropy per batch", loc="left")
        axes[i, 1].set_title(f"{s}: distinct predicted classes per batch", loc="left")
        for ax in axes[i]:
            ax.set_xlabel("batch"); ax.legend(loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(out / "health_per_batch.png", dpi=130); plt.close(fig)

    rs = [r for r in runs if "test" in r["pred"]]
    if rs:
        fig, axes = plt.subplots(len(rs), 1, figsize=(10, 2.2 * len(rs)), sharex=True, squeeze=False)
        for ax, r in zip(axes[:, 0], rs):
            h = np.bincount(r["pred"]["test"], minlength=C)
            ax.bar(range(C), h, color=color[r["name"]], width=0.7)
            ax.axhline(len(r["pred"]["test"]) / C, color=GRAY, ls="--", lw=1)
            ax.set_title(f"{r['name']}: test predictions per class (dashed: uniform)", loc="left")
        axes[-1, 0].set_xticks(range(C)); axes[-1, 0].set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=8)
        fig.tight_layout(); fig.savefig(out / "test_pred_hist.png", dpi=130); plt.close(fig)
    print(f"\nfigures written to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs written by run.py (optionally with --diag)")
    ap.add_argument("--seg", type=int, default=11, help="assumed val domain length in batches, for breakdowns")
    ap.add_argument("--bounds", default="", help="explicit val domain starts, e.g. 12,24 (overrides --seg)")
    ap.add_argument("--fig", default=str(ROOT / "runs" / "analysis"), help='figure dir; "" skips figures')
    args = ap.parse_args()
    runs = [load_run(d) for d in args.runs]
    ids = pd.read_csv(ROOT / "data" / "val" / "stream.csv")["id"].tolist()
    y = pd.read_csv(ROOT / "data" / "val" / "labels.csv").set_index("id").loc[ids, "label"].to_numpy()
    for r in runs:
        assert "val" not in r["ids"] or r["ids"]["val"] == ids, f"{r['name']}: val ids not in stream order"

    resources(runs)
    bounds = [int(b) for b in args.bounds.split(",")] if args.bounds else None
    acc, ncls, present, segs = val_section(runs, y, args.seg, bounds)
    label_free(runs, "val", acc=acc, true_ncls=ncls, chunks=segs)
    label_free(runs, "test")
    shift_section(runs, "val", acc=acc, present=present)
    shift_section(runs, "test")
    if args.fig:
        figures(runs, y, acc, ncls, segs, Path(args.fig))


if __name__ == "__main__":
    main()
