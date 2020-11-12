from CPRD.table import Patient,Practice,Clinical, Diagnosis, Hes
from CPRD.spark import read_txt, read_csv
import pyspark.sql.functions as F
from CPRD.utils import rename_col, check_dir, check_time, cvt_str2time
import os


def main(params, spark):
    # merge diagnoses from CPRD and HES, and map all codes to ICD10
    config = params['raw_file_path']

    # read data and process each table
    patient = Patient(read_txt(spark.sc, spark.sqlContext, path=config['patient']))\
        .accept_flag().yob_calibration().cvt_crd2date().cvt_tod2date().cvt_deathdate2date().get_pracid()\
        .drop('accept')
    practice = Practice(read_txt(spark.sc, spark.sqlContext, path=config['practice'])).cvt_lcd2date().cvt_uts2date()
    clinical = Clinical(read_txt(spark.sc, spark.sqlContext, path=config['clinical'])).rm_eventdate_medcode_empty()\
        .cvtEventDate2Time()

    # Filter clinical by time uts, crd, tod, lcd and get demographic data
    demographic = patient.join(practice, patient.pracid == practice.pracid, 'inner').drop('pracid')\
        .withColumn('startdate', F.greatest('uts', 'crd')).withColumn('enddate', F.least('tod', 'lcd'))
    time = demographic.select(['patid', 'startdate', 'enddate', 'deathdate'])
    clinical = clinical.join(time, time.patid == clinical.patid, 'inner').drop(time.patid).where(
        (F.col('eventdate') > F.col('startdate')) & (F.col('eventdate') < F.col('enddate'))).drop('deathdate')

    # Filter hes diagnosis data
    hes_diagnosis = rename_col(Diagnosis(
        read_txt(spark.sc, spark.sqlContext, path=config['diagnosis_hes'])).rm_date_icd_empty().cvt_admidate2date(),
                               old='admidate', new='eventdate')

    # merge hospital and clinical data
    clinical = rename_col(clinical.select(['patid', 'eventdate', 'medcode']).withColumn('source', F.lit('CPRD')),
                          'medcode', 'code')
    hes_diagnosis = rename_col(hes_diagnosis.select(['patid', 'eventdate', 'ICD']).withColumn('source', F.lit('hes')),
                               'ICD', 'code')
    diagnosis = clinical.union(hes_diagnosis)

    # filter by eligible patient table
    eligible = read_txt(spark.sc, spark.sqlContext, path=config['eligible']).where((F.col("hes_e") == 1))
    diagnosis = diagnosis.join(eligible, diagnosis.patid == eligible.patid, 'inner').drop(eligible.patid).select(
        ['patid', 'eventdate', 'source', 'code'])

    # filter by deathdate
    hes_death = Hes(
        read_txt(spark.sc, spark.sqlContext, path=config['hes_death']).select(['patid', 'dod'])).cvt_string2date('dod')
    diagnosis = diagnosis.join(hes_death, diagnosis.patid == hes_death.patid, 'left').drop(hes_death.patid)
    diagnosis_null = diagnosis.filter(F.col('dod').isNotNull() == False).drop('dod')
    diagnosis_death = diagnosis.filter(F.col('dod').isNotNull()).where(F.col('eventdate') < F.col('dod')).drop('dod')
    diagnosis = diagnosis_null.union(diagnosis_death)

    # map med code to read code
    med2read = read_txt(spark.sc, spark.sqlContext, config['med2read'])\
        .withColumn('medcode', F.col('medcode').cast('string')).select(['medcode', 'readcode'])

    # separate hes and cprd
    rm_dot = F.udf(lambda x: ''.join(x.split('.')))
    hes = diagnosis.filter(F.col('source') == 'hes').withColumn('code', rm_dot('code'))

    cprd = diagnosis.filter(F.col('source') == 'CPRD')
    cprd = cprd.join(med2read, cprd.code == med2read.medcode, 'left') \
        .select(['patid', 'eventdate', 'readcode', 'source']) \
        .filter(F.col('readcode').isNotNull()) \
        .withColumn('readcode', F.col('readcode').substr(0, 5)) \
        .withColumn('firstLetter', F.col('readcode').substr(0, 1))

    # clean up read and med code
    read2icd = read_csv(spark.sqlContext, config['icdNHS'])\
        .select(['read', 'icd']).withColumn('read', F.col('read').cast('string'))

    rm_x = F.udf(lambda x: x if x[-1] != 'X' else x[0:-1])
    read2icd = read2icd.withColumn('icd', rm_x('icd'))

    # noly map diagnose from read to icd
    cprd_non_diag = cprd.filter(F.col('firstLetter').isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Z', 'U'])) \
        .withColumn('type', F.lit('read')) \
        .withColumn('code', F.col('readcode')) \
        .select(['patid', 'eventdate', 'code', 'type', 'source'])

    cprd_diag = cprd.filter(
        F.col('firstLetter').isin(*['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Z', 'U']) == False
    )
    cprd_diag = cprd_diag.join(read2icd, cprd_diag.readcode == read2icd.read, 'left').withColumn('code', F.col('icd'))
    cprd_null = cprd_diag.filter(F.col('icd').isNotNull() == False)
    cprd_mapped = cprd_diag.filter(F.col('icd').isNotNull())

    # keep null code as read code
    cprd_null = cprd_null.withColumn('code', F.col('readcode')).withColumn('type', F.lit('read')).select(
        ['patid', 'eventdate', 'code', 'type', 'source'])
    cprd_mapped = cprd_mapped.withColumn('type', F.lit('icd')).select(['patid', 'eventdate', 'code', 'type', 'source'])
    cprd_diag = cprd_mapped.union(cprd_null).select(['patid', 'eventdate', 'code', 'type', 'source'])

    # merge cprd
    cprd = cprd_diag.union(cprd_non_diag)

    # union hes and cprd
    hes = hes.withColumn('type', F.lit('icd')).select(['patid', 'eventdate', 'code', 'type', 'source'])

    data = hes.union(cprd)
    data = check_time(data, 'eventdate', time_a=1985, time_b=2015)

    output = params['mid_stage_file_path']
    check_dir(output['mid_stage_dir'])

    data.write.parquet(os.path.join(output['mid_stage_dir'], output['mapped_diagnosis']))