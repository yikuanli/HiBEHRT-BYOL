from CPRD.table import Patient,Practice,Clinical, Diagnosis, Therapy, Hes
from CPRD.spark import read_txt, read_csv
import pyspark.sql.functions as F
from CPRD.utils import rename_col, check_dir, check_time
import os


def main(params, spark):
    # merge diagnoses from CPRD and HES, and map all codes to ICD10
    config = params['raw_file_path']

    # read data and process each table
    patient = Patient(read_txt(spark.sc, spark.sqlContext, path=config['patient'])) \
        .accept_flag().yob_calibration().cvt_crd2date().cvt_tod2date().cvt_deathdate2date().get_pracid().drop('accept')

    practice = Practice(
        read_txt(spark.sc, spark.sqlContext, path=config['practice'])).cvt_lcd2date().cvt_uts2date()

    therapy = Therapy(read_txt(spark.sc, spark.sqlContext, path=config['therapy'])) \
        .rm_eventdate_prodcode_empty().cvtEventDate2Time()

    demographic = patient.join(practice, patient.pracid == practice.pracid, 'inner').drop('pracid'). \
        withColumn('startdate', F.greatest('uts', 'crd')).withColumn('enddate', F.least('tod', 'lcd'))

    time = demographic.select(['patid', 'startdate', 'enddate', 'deathdate'])

    therapy = therapy.join(time, time.patid == therapy.patid, 'inner').drop(time.patid). \
        where((F.col('eventdate') > F.col('startdate')) & (F.col('eventdate') < F.col('enddate'))).drop('deathdate')

    # mapping from product to bnf
    extract_bnf = F.udf(lambda x: '/'.join([each[0:4] for each in x.split('/')]) if '/' in x else x[0:4])
    crossmap = read_txt(spark.sc, spark.sqlContext, path=config['prod2bnf']).select('prodcode', 'bnfcode') \
        .where((F.col("bnfcode") != '00000000')).withColumn('code', extract_bnf('bnfcode'))

    therapy = therapy.join(crossmap, therapy.prodcode == crossmap.prodcode, 'left').drop(crossmap.prodcode) \
        .dropDuplicates().select(['patid', 'eventdate', 'code']).dropna()

    therapy = check_time(therapy, 'eventdate', time_a=1985, time_b=2015)

    # remove none and duplicate
    output = params['mid_stage_file_path']
    check_dir(output['mid_stage_dir'])

    therapy.write.parquet(os.path.join(output['mid_stage_dir'], output['medication']))