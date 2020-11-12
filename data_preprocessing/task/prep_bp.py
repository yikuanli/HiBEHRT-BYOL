from CPRD.table import Patient,Practice,Clinical, Diagnosis, Therapy
from CPRD.spark import read_txt, read_csv, read_parquet
import pyspark.sql.functions as F
from CPRD.utils import rename_col, check_dir, check_time, cvt_str2time
import os


def main(params, spark):
    # merge diagnoses from CPRD and HES, and map all codes to ICD10
    config = params['mid_stage_file_path']

    additional = read_parquet(spark.sqlContext, os.path.join(config['mid_stage_dir'], config['additional']))

    bp = additional.filter(F.col('enttype') == 1).groupby(['patid', 'eventdate'])\
        .agg(F.first('data1').alias('bp_low'), F.first('data2').alias('bp_high'))\
        .dropna().select(['patid','eventdate','bp_low','bp_high'])

    bp_low = bp.select(['patid', 'eventdate', 'bp_low']).dropna() \
        .withColumn('bp_low', F.col('bp_low').cast('integer')).filter((F.col('bp_low') > 50) & (F.col('bp_low') < 140))
    bp_high = bp.select(['patid', 'eventdate', 'bp_high']).dropna() \
        .withColumn('bp_high', F.col('bp_high').cast('integer')).filter(
        (F.col('bp_high') > 80) & (F.col('bp_high') < 200))

    # remove none and duplicate
    output = params['mid_stage_file_path']
    check_dir(output['mid_stage_dir'])

    bp_low.write.parquet(os.path.join(output['mid_stage_dir'], output['bp_low']))
    bp_high.write.parquet(os.path.join(output['mid_stage_dir'], output['bp_high']))
    
