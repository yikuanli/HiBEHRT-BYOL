from CPRD.table import Patient, Practice, Clinical, Diagnosis, Therapy
from CPRD.spark import read_txt, read_csv, read_parquet
import pyspark.sql.functions as F
from CPRD.utils import rename_col, check_dir, check_time, cvt_str2time
import os


def main(params, spark):
    # merge diagnoses from CPRD and HES, and map all codes to ICD10
    config = params['mid_stage_file_path']

    additional = read_parquet(spark.sqlContext, os.path.join(config['mid_stage_dir'], config['additional']))

    smoke = additional.filter(F.col('enttype') == 4).groupby(['patid', 'eventdate']).agg(F.first('data1').alias('smoke'))\
    .select(['patid', 'eventdate', 'smoke'])

    # remove none and duplicate
    output = params['mid_stage_file_path']
    check_dir(output['mid_stage_dir'])

    smoke.write.parquet(os.path.join(output['mid_stage_dir'], output['smoke']))

