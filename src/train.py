"""Training loop.

Follows Sec. IV-B of the paper: BCE loss, Adam (beta1=0.9, beta2=0.999,
eps=1e-8, lr=1e-3), learning rate decayed by 0.1 when the loss plateaus for ten
validation passes, batch size 32 drawn in a "random but balanced" manner, and a
stop condition of either a plateau or the minimum learning rate of 1e-8.

Training is counted in ITERATIONS, matching Table IV.  The paper's 100k is only
a ceiling -- its own convergence figures are 27k-40.5k -- so --max-iters
defaults to the relevant Table IV entry plus 25% headroom.

Built for free Colab: checkpoints every --ckpt-every iterations and --resume
picks up exactly where a reclaimed session left off.

    python -m src.train --backbone xception --mode hfc --protocol identity \
        --faces-root faces --manifest faces/manifest.csv --splits splits.json \
        --out runs/xception_hfc_identity --resume
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import PAPER
from src.data.dataset import BalancedBatchSampler, CelebDFFaces
from src.eval import evaluate, format_metrics
from src.models.build import MODES, build_model, count_parameters
from src.models.hfc import FILTER_TYPES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, **state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)          # write-then-rename: a killed session
    tmp.replace(path)               # can never leave a half-written checkpoint


def build_loaders(args, device) -> tuple[DataLoader, DataLoader, CelebDFFaces]:
    common = dict(
        faces_root=args.faces_root, manifest_csv=args.manifest,
        splits_json=args.splits, protocol=args.protocol, size=args.image_size,
    )
    train_ds = CelebDFFaces(part="train", **common)
    val_ds = CelebDFFaces(part="val", return_meta=True, **common)
    print(f"  {train_ds.describe()}")
    print(f"  {val_ds.describe()}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_sampler=BalancedBatchSampler(train_ds, args.batch_size, seed=args.seed),
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=args.workers > 0,   # Windows/Colab spawn is expensive
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader, train_ds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train an HFC deepfake detector.")
    # data
    ap.add_argument("--faces-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--protocol", default="identity",
                    choices=["random", "identity", "identity_strict"])
    ap.add_argument("--image-size", type=int, default=PAPER.image_size)
    # model
    ap.add_argument("--backbone", default="xception")
    ap.add_argument("--mode", default="hfc", choices=list(MODES))
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--sigma", type=float, default=PAPER.gaussian_sigma)
    ap.add_argument("--kernel-size", type=int, default=PAPER.gaussian_kernel)
    ap.add_argument(
        "--literal-eq1", action="store_true",
        help="use the paper's printed 1/(2*pi*sigma^2) factor instead of a "
             "sum-to-1 kernel (see src/models/hfc.py)",
    )
    ap.add_argument(
        "--hfc-filter", default="eq1", choices=list(FILTER_TYPES),
        help="high-pass front end: 'eq1' is the paper's fixed Gaussian residual; "
             "'pyramid' is a multi-scale Laplacian band-pass; 'learned' trains N "
             "filters under the constrained-convolution constraint (centre 1, "
             "others summing to -1), of which Eq. (1) is one fixed member.",
    )
    ap.add_argument("--pyramid-levels", type=int, default=4)
    ap.add_argument("--n-filters", type=int, default=4)
    # optimisation
    ap.add_argument("--batch-size", type=int, default=PAPER.batch_size)
    ap.add_argument("--lr", type=float, default=PAPER.learning_rate)
    ap.add_argument("--max-iters", type=int, default=0, help="0 = Table IV budget")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument(
        "--select-by", default="auc", choices=["loss", "auc"],
        help="metric for choosing best.pt. Default 'auc' because that is what we "
             "report: BCE loss punishes confident-but-wrong predictions, so it can "
             "rise while ranking quality keeps improving, and selecting on loss then "
             "discards the better model.",
    )
    ap.add_argument("--patience", type=int, default=PAPER.lr_patience)
    ap.add_argument("--min-lr", type=float, default=PAPER.min_lr)
    ap.add_argument("--no-amp", action="store_true")
    # plumbing
    ap.add_argument("--out", required=True, help="checkpoint/log directory")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--overfit-batches", type=int, default=0,
        help="sanity check: train on N fixed batches and expect loss -> 0",
    )
    args = ap.parse_args(argv)

    if args.max_iters <= 0:
        args.max_iters = PAPER.budget_for(args.backbone, args.mode)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run_name = f"{args.backbone}/{args.mode}/{args.protocol}"
    if args.mode == "hfc" and args.hfc_filter != "eq1":
        run_name += f"[{args.hfc_filter}]"
    print(f"run={run_name}  device={device}  amp={use_amp}  max_iters={args.max_iters}")

    train_loader, val_loader, train_ds = build_loaders(args, device)

    model = build_model(
        args.backbone, args.mode,
        pretrained=not args.no_pretrained,
        kernel_size=args.kernel_size,
        sigma=args.sigma,
        normalize_kernel=not args.literal_eq1,
        filter_type=args.hfc_filter,
        pyramid_levels=args.pyramid_levels,
        n_filters=args.n_filters,
    ).to(device)
    n_params = count_parameters(model)
    expected = PAPER.paper_params_m.get(f"{args.backbone}/{args.mode}")
    note = f" (paper: {expected}M)" if expected else ""
    print(f"  params={n_params / 1e6:.3f}M{note}")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr,
        betas=PAPER.adam_betas, eps=PAPER.adam_eps,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PAPER.lr_decay_factor, patience=args.patience,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    config = {
        "backbone": args.backbone, "mode": args.mode, "protocol": args.protocol,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "sigma": args.sigma, "kernel_size": args.kernel_size,
        "normalize_kernel": not args.literal_eq1, "n_params": n_params,
        "hfc_filter": args.hfc_filter, "pyramid_levels": args.pyramid_levels,
        "n_filters": args.n_filters,
        "max_iters": args.max_iters,
    }

    start_iter, best_loss, best_iter, best_metrics = 0, float("inf"), 0, {}
    last_ckpt, best_ckpt = out / "last.pt", out / "best.pt"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_iter = ck["iteration"]
        best_loss, best_iter = ck["best_loss"], ck["best_iter"]
        best_metrics = ck.get("best_metrics", {})
        print(f"  resumed from iteration {start_iter}")

    log_path = out / "log.jsonl"
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # ---- sanity mode: can the loop drive a fixed set of batches to zero? ----
    if args.overfit_batches:
        fixed = [next(iter(train_loader)) for _ in range(args.overfit_batches)]
        model.train()
        for step in range(300):
            total = 0.0
            for imgs, labels in fixed:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    loss = criterion(model(imgs).squeeze(1).float(), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += loss.detach().item()
            mean = total / len(fixed)
            if step % 25 == 0:
                print(f"  overfit step {step:3d}  loss={mean:.5f}")
            if mean < 1e-3:
                print(f"OK: drove {args.overfit_batches} batch(es) to loss={mean:.2e}")
                return 0
        print(f"FAIL: loss stalled at {mean:.5f} -- check labels/sampler/LR")
        return 1

    # ---- main loop ---------------------------------------------------------
    train_iter = iter(train_loader)
    model.train()
    running, seen = 0.0, 0
    t0 = time.time()

    for it in range(start_iter + 1, args.max_iters + 1):
        imgs, labels = next(train_iter)
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            loss = criterion(model(imgs).squeeze(1).float(), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.detach().item()
        seen += 1

        if it % args.val_every == 0:
            # Expect the first few validation passes to look terrible, sometimes
            # with a huge loss: BatchNorm running statistics need a few hundred
            # iterations to converge, and until they do, eval-mode outputs are
            # badly calibrated even when the weights are already fine.  Judge a
            # run by its trend, not by its first validation.
            metrics = evaluate(model, val_loader, device, amp=use_amp)
            model.train()
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step(metrics["loss"])

            rate = seen / max(1e-9, time.time() - t0)
            print(
                f"  iter {it:>6d}/{args.max_iters}  train_loss={running / max(1, seen):.4f}  "
                f"val {format_metrics(metrics)}  lr={lr:.2e}  {rate:.1f} it/s",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "iteration": it,
                    "train_loss": running / max(1, seen),
                    "lr": lr,
                    "val": metrics,
                    "elapsed_s": round(time.time() - t0, 1),
                }) + "\n")
            running, seen, t0 = 0.0, 0, time.time()

            # Lower is better for loss, higher for AUC: score both as "lower wins".
            score = (
                metrics["loss"] if args.select_by == "loss"
                else -metrics["frame"]["auc"]
            )
            if score < best_loss:
                best_loss, best_iter, best_metrics = score, it, metrics
                save_checkpoint(
                    best_ckpt, model=model.state_dict(), config=config,
                    iteration=it, metrics=metrics,
                )

            if optimizer.param_groups[0]["lr"] < args.min_lr:
                print(f"  stopping: learning rate fell below {args.min_lr:g}")
                break

        if it % args.ckpt_every == 0 or it == args.max_iters:
            save_checkpoint(
                last_ckpt, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(), iteration=it, config=config,
                best_loss=best_loss, best_iter=best_iter, best_metrics=best_metrics,
            )

    summary = {
        "run": run_name,
        "config": config,
        "select_by": args.select_by,
        "best_iteration": best_iter,          # Table IV: iterations to convergence
        "best_val": best_metrics,
        "n_params": n_params,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    label = "val loss" if args.select_by == "loss" else "val frame AUC"
    shown = best_loss if args.select_by == "loss" else -best_loss
    print(f"\ndone: best {label} {shown:.4f} at iteration {best_iter}")
    print(f"  {format_metrics(best_metrics) if best_metrics else '(no validation ran)'}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
