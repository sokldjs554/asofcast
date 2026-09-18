"""Optional Spark profile for the UCI ElectricityLoadDiagrams20112014 dataset."""
from __future__ import annotations

import json
import time
from pathlib import Path


def measurement_cells(rows: int, client_columns: int) -> int:
    if rows < 1 or client_columns < 1:
        raise ValueError('positive row and client counts required')
    return int(rows) * int(client_columns)


def profile_uci_spark(input_path: Path, parquet_dir: Path, report_path: Path) -> dict:
    try:
        from pyspark.sql import SparkSession, functions as F
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyspark is required; install the 'spark' extra") from exc
    source = Path(input_path)
    if not source.is_file():
        raise ValueError('UCI source file is missing')
    started = time.perf_counter()
    spark = (SparkSession.builder.master('local[2]').appName('AsOfCast-UCI-profile')
             .config('spark.sql.shuffle.partitions', '4').getOrCreate())
    try:
        raw = spark.read.option('header', True).option('sep', ';').csv(str(source))
        if len(raw.columns) < 2:
            raise ValueError('UCI file must contain timestamp plus client columns')
        timestamp_column, clients = raw.columns[0], raw.columns[1:]
        typed = raw.select(F.to_timestamp(F.col(timestamp_column), 'yyyy-MM-dd HH:mm:ss').alias('event_time'),
            *[F.regexp_replace(F.col(name), ',', '.').cast('double').alias(name) for name in clients])
        rows = typed.count()
        if rows < 1:
            raise ValueError('UCI file contains no rows')
        if typed.filter(F.col('event_time').isNull()).limit(1).count():
            raise ValueError('unparseable UCI timestamps')
        null_expr = sum(F.when(F.col(name).isNull(), F.lit(1)).otherwise(F.lit(0)) for name in clients)
        if typed.select(null_expr.alias('nulls')).filter(F.col('nulls') > 0).limit(1).count():
            raise ValueError('numeric parsing produced nulls')
        parquet_dir = Path(parquet_dir)
        typed.write.mode('overwrite').parquet(str(parquet_dir))
        reread = spark.read.parquet(str(parquet_dir))
        parquet_rows = reread.count()
        if parquet_rows != rows:
            raise RuntimeError('Parquet row-count verification failed')
        sample_clients = clients[:min(8, len(clients))]
        means_row = reread.agg(*[F.mean(F.col(name)).alias(name) for name in sample_clients]).first()
        report = {'status':'verified','source':source.name,'rows':rows,'client_columns':len(clients),
                  'measurement_cells':measurement_cells(rows,len(clients)),'parquet_rows':parquet_rows,
                  'sample_client_means':{name:means_row[name] for name in sample_clients},
                  'spark_version':spark.version,'master':spark.sparkContext.master,
                  'duration_seconds':time.perf_counter()-started,
                  'scope':'full wide-table CSV parse, numeric cast, null scan, Parquet write/read; no long-form explode'}
        Path(report_path).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        return report
    finally:
        spark.stop()
