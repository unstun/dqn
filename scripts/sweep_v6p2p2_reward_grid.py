#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Combo:
    k_t: float
    k_delta: float


def _parse_floats(spec: str) -> list[float]:
    vals: list[float] = []
    for part in str(spec).split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("empty float list")
    return vals


def _slug(v: float) -> str:
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    return s.replace("-", "m").replace(".", "p")


def _run(cmd: list[str], *, dry_run: bool) -> int:
    print("$", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return int(proc.returncode)


def _latest_run_dir(out_name: str) -> Path | None:
    latest_txt = ROOT / "runs" / str(out_name) / "latest.txt"
    if not latest_txt.is_file():
        return None
    text = latest_txt.read_text(encoding="utf-8").strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = (latest_txt.parent / p).resolve()
    return p


def _to_float(x: str | None) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _mean(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals if math.isfinite(float(v))]
    if not arr:
        return float("nan")
    return float(sum(arr) / len(arr))


def _read_cnn_metrics(csv_path: Path) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = str(row.get("Algorithm", "")).strip().lower()
            if algo != "cnn-ddqn":
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No CNN-DDQN rows found in {csv_path}")

    envs = [str(r.get("Environment", "")).strip() for r in rows]
    success = [_to_float(r.get("success_rate")) for r in rows]
    path_len = [_to_float(r.get("avg_path_length")) for r in rows]
    curvature = [_to_float(r.get("avg_curvature_1_m")) for r in rows]
    path_time = [_to_float(r.get("path_time_s")) for r in rows]

    return {
        "suite_count": len(rows),
        "suites": ";".join(envs),
        "success_rate": _mean(success),
        "avg_path_length": _mean(path_len),
        "avg_curvature_1_m": _mean(curvature),
        "path_time_s": _mean(path_time),
    }


def _score_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    sr = float(row.get("success_rate", float("nan")))
    pl = float(row.get("avg_path_length", float("nan")))
    cv = float(row.get("avg_curvature_1_m", float("nan")))
    pt = float(row.get("path_time_s", float("nan")))
    return (-sr, pl, cv, pt)


def _default_fieldnames() -> list[str]:
    return [
        "stage",
        "status",
        "rank",
        "k_t",
        "k_delta",
        "seed",
        "profile",
        "train_out",
        "train_run_dir",
        "models_dir",
        "infer_out",
        "infer_run_dir",
        "kpi_csv",
        "suite_count",
        "suites",
        "success_rate",
        "avg_path_length",
        "avg_curvature_1_m",
        "path_time_s",
        "error",
    ]


def _write_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = _default_fieldnames()
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row_out = {k: r.get(k, "") for k in fields}
            w.writerow(row_out)


def _read_topk_from_csv(path: Path, topk: int) -> list[Combo]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("status", "")).strip().lower() != "ok":
                continue
            rows.append(row)

    def rank_of(r: dict[str, str]) -> int:
        try:
            return int(float(str(r.get("rank", "")).strip()))
        except Exception:
            return 10**9

    rows.sort(key=rank_of)
    combos: list[Combo] = []
    seen: set[tuple[float, float]] = set()
    for r in rows:
        try:
            kt = float(str(r.get("k_t", "")))
            kd = float(str(r.get("k_delta", "")))
        except Exception:
            continue
        key = (kt, kd)
        if key in seen:
            continue
        seen.add(key)
        combos.append(Combo(k_t=kt, k_delta=kd))
        if len(combos) >= int(topk):
            break
    return combos


def _build_grid(k_t_values: list[float], k_delta_values: list[float]) -> list[Combo]:
    combos: list[Combo] = []
    for kt in k_t_values:
        for kd in k_delta_values:
            combos.append(Combo(k_t=float(kt), k_delta=float(kd)))
    return combos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v6p2p2 reward grid sweep for forest CNN-DDQN")
    ap.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    ap.add_argument(
        "--smoke-profile",
        default="repro_20260219_v6p2p2_reward_kt_kdelta_sweep_smoke",
        help="Profile for smoke stage.",
    )
    ap.add_argument(
        "--full-profile",
        default="repro_20260219_v6p2p2_reward_kt_kdelta_sweep_full",
        help="Profile for full stage.",
    )
    ap.add_argument("--k-t-values", default="0.06,0.08,0.1,0.12,0.15")
    ap.add_argument("--k-delta-values", default="0.8,1.2,1.5,2.0,2.5")
    ap.add_argument("--topk", type=int, default=3, help="Only used by full stage.")
    ap.add_argument(
        "--candidates-csv",
        default="",
        help="Optional CSV from smoke summary. If omitted in full stage, use latest smoke summary.",
    )
    ap.add_argument("--out-prefix", default="repro_20260219_v6p2p2_reward_sweep")
    ap.add_argument("--summary-dir", default="runs/repro_20260219_v6p2p2_reward_sweep")
    ap.add_argument("--seed-base", type=int, default=2100)
    ap.add_argument("--max-combos", type=int, default=0, help="0 means all combos.")
    ap.add_argument("--conda-env", default="ros2py310")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    summary_dir = (ROOT / str(args.summary_dir)).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "smoke":
        combos = _build_grid(_parse_floats(args.k_t_values), _parse_floats(args.k_delta_values))
    else:
        if args.candidates_csv:
            candidate_csv = Path(args.candidates_csv)
            if not candidate_csv.is_absolute():
                candidate_csv = (ROOT / candidate_csv).resolve()
        else:
            candidate_csv = summary_dir / "smoke_summary_latest.csv"
        if not candidate_csv.is_file():
            raise FileNotFoundError(f"Candidates CSV not found: {candidate_csv}")
        combos = _read_topk_from_csv(candidate_csv, topk=max(1, int(args.topk)))
        if not combos:
            raise RuntimeError(f"No valid candidates found in {candidate_csv}")

    if int(args.max_combos) > 0:
        combos = combos[: int(args.max_combos)]

    if not combos:
        raise RuntimeError("No combos to run")

    profile = str(args.smoke_profile if args.stage == "smoke" else args.full_profile)

    rows: list[dict[str, object]] = []
    for i, combo in enumerate(combos):
        kt = float(combo.k_t)
        kd = float(combo.k_delta)
        seed = int(args.seed_base) + int(i)
        tag = f"kt{_slug(kt)}_kd{_slug(kd)}"
        train_out = f"{args.out_prefix}_{args.stage}_{tag}_train"
        infer_out = f"{args.out_prefix}_{args.stage}_{tag}_infer"

        row: dict[str, object] = {
            "stage": str(args.stage),
            "status": "pending",
            "rank": "",
            "k_t": kt,
            "k_delta": kd,
            "seed": seed,
            "profile": profile,
            "train_out": train_out,
            "train_run_dir": "",
            "models_dir": "",
            "infer_out": infer_out,
            "infer_run_dir": "",
            "kpi_csv": "",
            "suite_count": "",
            "suites": "",
            "success_rate": "",
            "avg_path_length": "",
            "avg_curvature_1_m": "",
            "path_time_s": "",
            "error": "",
        }

        train_cmd = [
            "conda",
            "run",
            "-n",
            str(args.conda_env),
            "python",
            "train.py",
            "--profile",
            profile,
            "--out",
            train_out,
            "--seed",
            str(seed),
            "--forest-reward-k-t",
            str(kt),
            "--forest-reward-k-delta",
            str(kd),
        ]
        rc = _run(train_cmd, dry_run=bool(args.dry_run))
        if rc != 0:
            row["status"] = "train_failed"
            row["error"] = f"train rc={rc}"
            rows.append(row)
            continue

        train_run_dir = _latest_run_dir(train_out)
        if train_run_dir is None:
            row["status"] = "train_missing_latest"
            row["error"] = "runs/<out>/latest.txt not found"
            rows.append(row)
            continue

        models_dir = train_run_dir / "models"
        row["train_run_dir"] = str(train_run_dir)
        row["models_dir"] = str(models_dir)

        infer_cmd = [
            "conda",
            "run",
            "-n",
            str(args.conda_env),
            "python",
            "infer.py",
            "--profile",
            profile,
            "--models",
            str(models_dir),
            "--out",
            infer_out,
            "--seed",
            str(seed),
        ]
        rc = _run(infer_cmd, dry_run=bool(args.dry_run))
        if rc != 0:
            row["status"] = "infer_failed"
            row["error"] = f"infer rc={rc}"
            rows.append(row)
            continue

        infer_run_dir = _latest_run_dir(infer_out)
        if infer_run_dir is None:
            row["status"] = "infer_missing_latest"
            row["error"] = "runs/<out>/latest.txt not found"
            rows.append(row)
            continue

        kpi_csv = infer_run_dir / "table2_kpis_mean_raw.csv"
        row["infer_run_dir"] = str(infer_run_dir)
        row["kpi_csv"] = str(kpi_csv)

        try:
            metrics = _read_cnn_metrics(kpi_csv)
            row.update(metrics)
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "metric_parse_failed"
            row["error"] = str(exc)

        rows.append(row)

    ok_rows = [r for r in rows if str(r.get("status", "")) == "ok"]
    ok_rows.sort(key=_score_key)
    for rank, r in enumerate(ok_rows, start=1):
        r["rank"] = rank

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = summary_dir / f"{args.stage}_summary_{timestamp}.csv"
    _write_csv(rows, out_csv)

    latest_csv = summary_dir / f"{args.stage}_summary_latest.csv"
    shutil.copyfile(out_csv, latest_csv)

    print(f"[summary] wrote: {out_csv}")
    print(f"[summary] latest: {latest_csv}")
    print("[summary] top candidates:")
    for r in ok_rows[: max(1, int(args.topk))]:
        print(
            "  "
            + f"rank={r.get('rank')} k_t={r.get('k_t')} k_delta={r.get('k_delta')} "
            + f"sr={r.get('success_rate')} len={r.get('avg_path_length')} "
            + f"curv={r.get('avg_curvature_1_m')} t={r.get('path_time_s')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
