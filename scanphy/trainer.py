"""Model training (PyTorch). Runs on CUDA (Jetson Orin / desktop GPU) or CPU.

Two typical uses:

  1. Base model (already shipped in models/scanphy-base.npz), trained on synthetic,
     domain-randomised rooms:
         python -m scanphy train --synthetic 120000 --out models/scanphy-base.npz

  2. Room-specific fine-tuning on the Jetson, using recordings captured with the
     dashboard's calibration wizard (this is what auto-train runs):
         python -m scanphy train --recordings data/recordings --config config/room.json \
             --init models/scanphy-base.npz --out models/room-<date>.npz

Progress is printed as JSON lines when --json-progress is given (consumed by the
dashboard's training panel).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Optional

import numpy as np

from . import config as cfgmod
from .dataset import load_all
from .features import ACTIVITIES, FEATURE_DIM
from .model import LAYER_KEYS, save_npz
from .simulator import synth_dataset

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - handled at runtime
    torch = None


def emit(progress: bool, **kw) -> None:
    if progress:
        print(json.dumps(kw), flush=True)
    else:
        if kw.get("event") == "epoch":
            v = kw.get("val", {})
            print("epoch %3d/%d  loss %.4f  act %.3f  zone %.3f  xy %.2fm  lr %.2e" % (
                kw["epoch"], kw["epochs"], kw["loss"], v.get("activity_acc", 0), v.get("zone_acc", 0),
                v.get("xy_mae_m", 0), kw.get("lr", 0)), flush=True)
        elif kw.get("event") in ("log", "done", "error"):
            print("[%s] %s" % (kw["event"], kw.get("message", "")), flush=True)


if torch is not None:
    class Net(nn.Module):
        def __init__(self, d_in: int, n_zone: int, hidden: int = 256, dropout: float = 0.1):
            super().__init__()
            self.l0 = nn.Linear(d_in, hidden)
            self.n0 = nn.LayerNorm(hidden)
            self.l1 = nn.Linear(hidden, hidden)
            self.n1 = nn.LayerNorm(hidden)
            self.l2 = nn.Linear(hidden, 128)
            self.act = nn.Linear(128, len(ACTIVITIES))
            self.zone = nn.Linear(128, n_zone)
            self.xy = nn.Linear(128, 2)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = F.relu(self.n0(self.l0(x)))
            h = self.drop(h)
            h = F.relu(self.n1(self.l1(h)))
            h = self.drop(h)
            h = F.relu(self.l2(h))
            return self.act(h), self.zone(h), torch.sigmoid(self.xy(h))

    class Exportable(nn.Module):
        """Normalisation + softmax baked in, for ONNX / TensorRT export."""

        def __init__(self, net, mu, sigma):
            super().__init__()
            self.net = net
            self.register_buffer("mu", torch.tensor(mu))
            self.register_buffer("sigma", torch.tensor(sigma))

        def forward(self, x):
            a, z, xy = self.net((x - self.mu) / self.sigma)
            return torch.softmax(a, -1), torch.softmax(z, -1), xy


def _tkey(k: str) -> str:
    """npz key ('l0.w') -> torch state_dict key ('l0.weight')."""
    return k[:-2] + (".weight" if k.endswith(".w") else ".bias")


def _load_init(net, path: str):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        sd = net.state_dict()
        for k in LAYER_KEYS:
            if tuple(z[k].shape) != tuple(sd[_tkey(k)].shape):
                continue  # e.g. zone grid changed: keep that head randomly initialised
            sd[_tkey(k)] = torch.tensor(z[k])
        net.load_state_dict(sd)
        return z["mu"].astype(np.float32), z["sigma"].astype(np.float32), meta


def _metrics(net, X, act, xy, hx, zone, dims, device, batch=4096) -> Dict[str, float]:
    net.eval()
    pa, pz, pxy = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            a, z, p = net(torch.tensor(X[i:i + batch], device=device))
            pa.append(a.argmax(-1).cpu().numpy())
            pz.append(z.argmax(-1).cpu().numpy())
            pxy.append(p.cpu().numpy())
    pa, pz, pxy = np.concatenate(pa), np.concatenate(pz), np.concatenate(pxy)
    person = (act > 0) & hx
    out = {"activity_acc": float((pa == act).mean()),
           "presence_acc": float(((pa > 0) == (act > 0)).mean())}
    if person.any():
        err = (pxy[person] - xy[person]) * dims[person]
        out["zone_acc"] = float((pz[person] == zone[person]).mean())
        out["xy_mae_m"] = float(np.linalg.norm(err, axis=1).mean())
    else:
        out["zone_acc"], out["xy_mae_m"] = 0.0, 0.0
    return out


def train(args) -> int:
    progress = args.json_progress
    if torch is None:
        emit(progress, event="error", message="PyTorch is not installed. On Jetson run scripts/install_torch_jetson.sh")
        return 2
    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dev_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"
    emit(progress, event="log", message="training on %s" % dev_name)

    cfg = cfgmod.load_config(args.config) if args.config else None
    rows = cfg["zones"]["rows"] if cfg else 3
    cols = cfg["zones"]["cols"] if cfg else 3
    if cfg is None and args.recordings:
        emit(progress, event="log", message="no --config given: using default 3x3 zone grid")

    # ---- data ---------------------------------------------------------------------------
    parts = []  # (X, act, xy, hx, zone, dims, weight, group, is_real)
    if args.synthetic > 0:
        emit(progress, event="log", message="generating %d synthetic windows%s" % (
            args.synthetic, " (randomised rooms + your room layout)" if cfg else " (randomised rooms)"))
        geo_cfg = cfg
        n_generic = args.synthetic if geo_cfg is None else args.synthetic // 3
        n_room = 0 if geo_cfg is None else args.synthetic - n_generic
        for n, c, seed in ((n_generic, None, args.seed), (n_room, geo_cfg, args.seed + 1)):
            if n <= 0:
                continue
            X, act, xy, zone, dims = synth_dataset(n, seed=seed, cfg=c, workers=args.workers)
            if c is None and (rows, cols) != (3, 3):
                zone = np.array([cfgmod.zone_index(a, b, rows, cols) for a, b in xy], dtype=np.int64)
            parts.append((X, act, xy, np.ones(len(X), bool), zone, dims, np.full(len(X), 1.0, np.float32),
                          np.random.randint(0, 10, size=len(X)), False))
    n_real = 0
    if args.recordings:
        real = load_all(args.recordings, window_s=1.0, hop_s=args.hop)
        if real is None:
            emit(progress, event="log", message="no usable recordings in %s" % args.recordings)
        else:
            X, act, xy, hx, zone, dims, groups = real
            if cfg is not None:
                zone = np.array([cfgmod.zone_index(a, b, rows, cols) for a, b in xy], dtype=np.int64)
            n_real = len(X)
            parts.append((X, act, xy, hx, zone, dims, np.full(len(X), args.real_weight, np.float32),
                          groups, True))
            emit(progress, event="log", message="%d windows from %d recordings" % (n_real, len(set(groups.tolist()))))
    if not parts:
        emit(progress, event="error", message="no training data (use --synthetic N and/or --recordings DIR)")
        return 1

    def split(p):
        X, act, xy, hx, zone, dims, w, g, is_real = p
        if is_real:
            ug = np.unique(g)
            if len(ug) >= 5:
                rng = np.random.default_rng(args.seed)
                val_g = rng.choice(ug, size=max(1, len(ug) // 5), replace=False)
                vm = np.isin(g, val_g)
            else:  # too few recordings: hold out the tail of every recording
                vm = np.zeros(len(X), bool)
                for gg in ug:
                    idx = np.where(g == gg)[0]
                    vm[idx[int(len(idx) * 0.8):]] = True
        else:
            vm = np.random.default_rng(args.seed).random(len(X)) < 0.08
        tr = tuple(a[~vm] for a in (X, act, xy, hx, zone, dims, w))
        va = tuple(a[vm] for a in (X, act, xy, hx, zone, dims))
        return tr, va, is_real

    splits = [split(p) for p in parts]
    tr = [np.concatenate([s[0][j] for s in splits]) for j in range(7)]
    val_sets = {("real" if s[2] else "synthetic"): s[1] for s in splits if len(s[1][0])}
    Xtr, Atr, XYtr, HXtr, Ztr, Dtr, Wtr = tr

    net = Net(FEATURE_DIM, rows * cols).to(device)
    init_meta = None
    if args.init and os.path.exists(args.init):
        mu, sigma, init_meta = _load_init(net, args.init)
        emit(progress, event="log", message="fine-tuning from %s" % os.path.basename(args.init))
    else:
        mu = Xtr.mean(axis=0).astype(np.float32)
        sigma = (Xtr.std(axis=0) + 1e-3).astype(np.float32)
    norm = lambda X: ((X - mu) / sigma).astype(np.float32)  # noqa: E731
    Xtr_n = torch.tensor(norm(Xtr), device=device)
    t_act = torch.tensor(Atr, device=device)
    t_xy = torch.tensor(XYtr, device=device)
    t_hx = torch.tensor((HXtr & (Atr > 0)).astype(np.float32), device=device)
    t_zone = torch.tensor(Ztr, device=device)
    t_w = torch.tensor(Wtr, device=device)
    val_n = {k: (norm(v[0]),) + tuple(v[1:]) for k, v in val_sets.items()}

    lr = args.lr if args.lr else (3e-4 if args.init else 2e-3)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, math.ceil(len(Xtr_n) / args.batch))
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.15)
    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(Xtr_n), device=device)
        tot, cnt = 0.0, 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            x = Xtr_n[idx]
            x = x + 0.03 * torch.randn_like(x)  # light feature noise for robustness
            la, lz, pxy = net(x)
            w = t_w[idx]
            m = t_hx[idx]
            loss_a = (F.cross_entropy(la, t_act[idx], reduction="none", label_smoothing=0.05) * w).mean()
            loss_z = (F.cross_entropy(lz, t_zone[idx], reduction="none") * w * m).sum() / (m.sum() + 1e-6)
            loss_xy = ((F.smooth_l1_loss(pxy, t_xy[idx], reduction="none", beta=0.05).sum(-1)) * w * m).sum() \
                / (m.sum() + 1e-6)
            loss = loss_a + 0.5 * loss_z + 6.0 * loss_xy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        vals = {k: _metrics(net, *v, device=device) for k, v in val_n.items()}
        main = vals.get("real") or vals.get("synthetic") or {}
        rec = {"epoch": epoch, "loss": tot / max(cnt, 1), "val": main, "val_all": vals,
               "lr": sched.get_last_lr()[0]}
        history.append({"epoch": epoch, "loss": rec["loss"], **{k: v for k, v in main.items()}})
        emit(progress, event="epoch", epochs=args.epochs, **rec)
        score = main.get("activity_acc", 0) + main.get("zone_acc", 0) - 0.2 * main.get("xy_mae_m", 0)
        if best is None or score >= best[0]:
            best = (score, {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}, vals, epoch)

    _, state, vals, best_epoch = best
    meta = {
        "kind": "room" if n_real else ("layout" if cfg else "generic"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "zones": {"rows": rows, "cols": cols},
        "metrics": vals,
        "best_epoch": best_epoch,
        "samples": {"real": int(n_real), "total": int(len(Xtr) + sum(len(v[0]) for v in val_sets.values()))},
        "device": dev_name,
        "train_seconds": round(time.time() - t_start, 1),
        "history": history,
        "parent": os.path.basename(args.init) if args.init else None,
        "room": cfg["room"] if cfg else None,
        "layout": {"router": cfg["router"], "nodes": cfg["nodes"]} if cfg else None,
    }
    params = {k: state[_tkey(k)] for k in LAYER_KEYS}
    save_npz(args.out, params, mu, sigma, meta)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    if args.onnx:
        try:
            net.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
            exp = Exportable(net.cpu().eval(), mu, sigma)
            onnx_path = os.path.splitext(args.out)[0] + ".onnx"
            torch.onnx.export(exp, torch.zeros(1, FEATURE_DIM), onnx_path, input_names=["features"],
                              output_names=["activity", "zone", "xy"],
                              dynamic_axes={"features": {0: "n"}}, opset_version=17, dynamo=False)
            emit(progress, event="log", message="ONNX exported: %s" % onnx_path)
        except Exception as exc:  # ONNX is optional
            emit(progress, event="log", message="ONNX export skipped (%s)" % exc)
    emit(progress, event="done", message="saved %s" % args.out, path=args.out, metrics=vals,
         seconds=meta["train_seconds"])
    return 0


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--synthetic", type=int, default=30000, help="number of synthetic windows (0 = none)")
    p.add_argument("--recordings", default=None, help="directory of recordings from the dashboard")
    p.add_argument("--config", default=None, help="room config (geometry + zone grid)")
    p.add_argument("--init", default=None, help="start from this .npz model (fine-tuning)")
    p.add_argument("--out", default="models/scanphy-room.npz")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--hop", type=float, default=0.2, help="window hop (s) when slicing recordings")
    p.add_argument("--real-weight", type=float, default=3.0, help="loss weight of real vs synthetic windows")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--onnx", action="store_true", help="also export ONNX (for TensorRT)")
    p.add_argument("--json-progress", action="store_true")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="scanphy train")
    add_args(p)
    return train(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
