from CPRD.table import Patient,Practice,Clinical, Diagnosis, Therapy, Hes
from CPRD.spark import read_txt, read_csv
import pyspark.sql.functions as F
from CPRD.utils import rename_col, check_dir, check_time, cvt_str2time
import os


def main(params, spark):
    # merge diagnoses from CPRD and HES, and map all codes to ICD10
    config = params['raw_file_path']

    test = read_txt(spark.sc, spark.sqlContext, config['test']).drop('staffid').drop('sysdate')
    med2read = read_txt(spark.sc, spark.sqlContext, config['med2read']).withColumn('medcode', F.col('medcode').cast(
        'string')).select(['medcode', 'readcode'])

    test = test.join(med2read, test.medcode == med2read.medcode, 'left') \
        .filter(F.col('readcode').isNotNull()) \
        .withColumn('readcode', F.col('readcode').substr(0, 5)) \
        .select(['patid', 'eventdate', 'readcode', 'enttype', 'data1', 'data2'])

    test = test.withColumn('eventdate', cvt_str2time(test, 'eventdate')).dropDuplicates()

    test = test.filter(F.col('eventdate').isNotNull())

    test = check_time(test, 'eventdate', time_a=1985, time_b=2015)

    # remove none and duplicate
    output = params['mid_stage_file_path']
    check_dir(output['mid_stage_dir'])

    test.write.parquet(os.path.join(output['mid_stage_dir'], output['cprd_test']))