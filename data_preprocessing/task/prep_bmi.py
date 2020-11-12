from CPRD.table import Patient,Practice,Clinical, Diagnosis, Therapy
from CPRD.spark import read_txt, read_csv, read_parquet
import pyspark.sql.functions as F
from CPRD.utils import rename_col, check_dir, check_time, cvt_str2time
import os


def main(params, spark):
    # merge diagnoses from CPRD and HES, and map all codes to ICD10
    config = params['mid_stage_file_path']

    additional = read_parquet(spark.sqlContext, os.path.join(config['mid_stage_dir'], config['additional']))

    bmi = additional.filter(F.col('enttype') == 13).groupby(['patid', 'eventdate']).agg(F.mean('data3').alias('BMI'))\
        .dropna().where((F.col('BMI')>16)&(F.col('BMI')<50)).select(['patid','eventdate','BMI'])

    # remove none and duplicate
    output = params['mid_stage_file_path']
    check_dir(output['mid_stage_dir'])

    bmi.write.parquet(os.path.join(output['mid_stage_dir'], output['bmi']))