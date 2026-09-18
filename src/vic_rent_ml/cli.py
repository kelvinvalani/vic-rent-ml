"""Reproducible ingestion, selection, training and serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import uvicorn

from .ingest import build_panel_from_transformed, build_panel_from_workbook
from .serving import create_app
from .training import evaluate, train


def cmd_ingest(args):
    mapping = json.loads(Path(args.postcodes).read_text()) if args.postcodes else {}
    regions = pd.read_csv(args.regions, dtype={"Postcode": str}) if args.regions else pd.DataFrame()
    region_map = dict(zip(regions["Postcode"], regions["Region"])) if not regions.empty else {}
    panel = (
        build_panel_from_workbook(Path(args.rent), mapping)
        if args.rent else build_panel_from_transformed(Path(args.transformed), region_map)
    )
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(dest, index=False)
    print(f"Wrote {len(panel):,} modelling rows to {dest}")


def cmd_train(args):
    out = Path(args.out)
    payload = train(pd.read_csv(args.panel), out.parent, reuse_selection=args.reuse_selection)
    if out.name != "production_clone.joblib":
        joblib.dump(payload, out, compress=3)
    print(f"Saved {payload['recipe']} to {out}")


def cmd_evaluate(args):
    winner = evaluate(pd.read_csv(args.panel), Path(args.out_dir))
    print(json.dumps(winner, indent=2))


def cmd_check_regression(args):
    manifest = json.loads(Path(args.manifest).read_text())
    winner = manifest["recipe"]
    row = next(item for item in manifest["test_metrics"] if item["model"] == winner)
    if row["test_mae"] > args.threshold_mae:
        raise SystemExit(f"Test MAE {row['test_mae']:.3f} exceeds {args.threshold_mae:.3f}")
    print(f"Recorded audit MAE {row['test_mae']:.3f} meets threshold {args.threshold_mae:.3f}")


def cmd_serve(args):
    app = create_app(joblib.load(args.model))
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(prog="vic-rent-ml")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Parse official moving-annual rent workbook")
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--rent")
    source.add_argument("--transformed", help="Existing application historical CSV")
    ingest.add_argument("--postcodes", help="Optional suburb-group to representative-postcode JSON")
    ingest.add_argument("--regions", help="Optional Postcode,Region CSV for regional serving fallback")
    ingest.add_argument("--out", default="data/rent_panel.csv")
    ingest.set_defaults(func=cmd_ingest)
    fitting = commands.add_parser("train", help="Select, calibrate, audit and refit the winner")
    fitting.add_argument("--panel", default="data/rent_panel.csv")
    fitting.add_argument("--out", default="artifacts/production_clone.joblib")
    fitting.add_argument("--reuse-selection", action="store_true", help="Validate and reuse saved validation results")
    fitting.set_defaults(func=cmd_train)
    evaluation = commands.add_parser("evaluate", help="Validation bake-off only")
    evaluation.add_argument("--panel", required=True)
    evaluation.add_argument("--out-dir", default="artifacts")
    evaluation.set_defaults(func=cmd_evaluate)
    check = commands.add_parser("check-regression", help="Check recorded audit metrics")
    check.add_argument("--manifest", required=True)
    check.add_argument("--threshold-mae", type=float, required=True)
    check.set_defaults(func=cmd_check_regression)
    serve = commands.add_parser("serve", help="Serve recursive forecasts over HTTP")
    serve.add_argument("--model", default="artifacts/production_clone.joblib")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
