from CPRD.table import Patient,Practice,Clinical, Diagnosis, Therapy
from CPRD.spark import read_txt, read_csv
import pyspark.sql.functions as F
from CPRD.utils import rename_col, check_dir, check_time, cvt_str2time
import os


def main(params, spark):
    # merge diagnoses from CPRD and HES, and map all codes to ICD10
    config = params['raw_file_path']

    additional = read_txt(spark.sc, spark.sqlContext, config['additional'])
    clinical = Clinical(read_txt(spark.sc, spark.sqlContext, path=config['clinical'])) \
        .rm_eventdate_medcode_empty().cvtEventDate2Time() \
        .select(['patid', 'eventdate', 'adid'])

    clinical = clinical.join(additional, (clinical.patid == additional.patid) & (clinical.adid == additional.adid)) \
        .drop(additional.patid).drop(additional.adid) \
        .select(['patid', 'eventdate', 'enttype', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']) \
        .dropDuplicates()

    clinical = check_time(clinical, 'eventdate', time_a=1985, time_b=2015)

    # remove none and duplicate
    output = params['mid_stage_file_path']
    check_dir(output['mid_stage_dir'])

    clinical.write.parquet(os.path.join(output['mid_stage_dir'], output['additional']))