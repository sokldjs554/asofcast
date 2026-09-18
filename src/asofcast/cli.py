"""CLI keeps generated test data separate from real-data and optional-stack verification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='asofcast',description='Point-in-time forecasting and learned waiting')
    commands = parser.add_subparsers(dest='command',required=True)
    fetch = commands.add_parser('fetch-ett',help='Fetch the checksum-pinned original ETT CSV')
    fetch.add_argument('--out',type=Path,default=Path('data/raw/ETTh1.csv'))
    generate = commands.add_parser('generate',help='Explicitly generate synthetic pipeline-test measurements')
    generate.add_argument('--out',type=Path,default=Path('data/synthetic/demo.csv'))
    generate.add_argument('--rows',type=int,default=7200)
    generate.add_argument('--seed',type=int,default=21)
    run = commands.add_parser('run',help='Train, evaluate and export a verified model bundle')
    run.add_argument('--csv',type=Path,required=True)
    run.add_argument('--out',type=Path,required=True)
    run.add_argument('--config',type=Path)
    run.add_argument('--source-kind',choices=['synthetic','ett','user-provided-csv'],required=True)
    verify = commands.add_parser('verify',help='Check bundle hashes and load weights safely')
    verify.add_argument('--artifacts',type=Path,default=Path('artifacts/demo'))
    benchmark = commands.add_parser('benchmark',help='Measure actual FP32 and dynamic INT8 CPU inference')
    benchmark.add_argument('--artifacts',type=Path,default=Path('artifacts/demo'))
    benchmark.add_argument('--out',type=Path,default=Path('runtime_benchmark.json'))
    benchmark.add_argument('--repeats',type=int,default=300)
    onnx_eval = commands.add_parser('onnx-eval',help='Export and compare ONNX Runtime against PyTorch')
    onnx_eval.add_argument('--artifacts',type=Path,required=True)
    onnx_eval.add_argument('--out',type=Path,required=True)
    onnx_eval.add_argument('--repeats',type=int,default=200)
    mlflow_cmd = commands.add_parser('track-mlflow',help='Log a verified bundle to an MLflow tracking store')
    mlflow_cmd.add_argument('--artifacts',type=Path,required=True)
    mlflow_cmd.add_argument('--tracking-uri',default='sqlite:///mlflow.db')
    mlflow_cmd.add_argument('--experiment',default='AsOfCast')
    spark = commands.add_parser('spark-profile',help='Process UCI ElectricityLoadDiagrams with Spark and Parquet')
    spark.add_argument('--input',type=Path,required=True)
    spark.add_argument('--parquet',type=Path,required=True)
    spark.add_argument('--out',type=Path,required=True)
    serve = commands.add_parser('serve',help='Run the local model-backed web service')
    serve.add_argument('--artifacts',type=Path,default=Path('artifacts/demo'))
    serve.add_argument('--host',default='127.0.0.1')
    serve.add_argument('--port',type=int,default=8000)
    args = parser.parse_args(argv)
    try:
        if args.command == 'generate':
            from asofcast.data import write_synthetic_csv
            result = write_synthetic_csv(args.out,n=args.rows,seed=args.seed)
        elif args.command == 'fetch-ett':
            from asofcast.data import download_ett
            result = download_ett(args.out)
        elif args.command == 'run':
            from asofcast.experiment import run_experiment
            cfg = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
            result = run_experiment(args.csv,args.out,cfg,source_kind=args.source_kind)
        elif args.command == 'verify':
            from asofcast.bundle import load_bundle
            bundle = load_bundle(args.artifacts)
            result = {'status':'verified','run_id':bundle.report['run_id'],'source_kind':bundle.report['source']['kind']}
        elif args.command == 'benchmark':
            from asofcast.optimize import benchmark_bundle
            result = benchmark_bundle(args.artifacts,args.out,repeats=args.repeats)
        elif args.command == 'onnx-eval':
            from asofcast.onnx_eval import evaluate_onnx
            result = evaluate_onnx(args.artifacts,args.out,repeats=args.repeats)
        elif args.command == 'track-mlflow':
            from asofcast.mlops import log_bundle_mlflow
            result = log_bundle_mlflow(args.artifacts,args.tracking_uri,args.experiment)
        elif args.command == 'spark-profile':
            from asofcast.large_data import profile_uci_spark
            result = profile_uci_spark(args.input,args.parquet,args.out)
        else:
            import uvicorn
            from asofcast.service import create_app
            uvicorn.run(create_app(args.artifacts),host=args.host,port=args.port,log_level='info')
            return 0
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    except (OSError,ValueError,RuntimeError) as exc:
        print(f'AsOfCast: {exc}',file=sys.stderr)
        return 2
