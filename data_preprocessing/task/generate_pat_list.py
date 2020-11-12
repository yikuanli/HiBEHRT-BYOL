from CPRD.table import Patient,Practice,Clinical, Diagnosis, Therapy, EHR
from CPRD.spark import read_txt, read_csv, read_parquet
import pyspark.sql.functions as F
from CPRD.utils import rename_col, check_dir, check_time, cvt_str2time
import os
from pyspark.sql import Window
from pyspark.sql.types import *
import random
import numpy as np
import datetime


def main(params, spark):
    global_params = {
        'min_visit': 5,
        'span': 12,
        'code_col': 'code',
        'hf_list': ['I099', 'I110', 'I130', 'I132', 'I255', 'I279', 'I38',
                    'I420', 'I421', 'I422', 'I426', 'I428', 'I429', 'I500',
                    'I501', 'I502', 'I503', 'I508', 'I509'],
        'start': 2005,
        'end': 2015
    }

    low_bound = 30 * 6  # 6 minths
    high_bound = 365  # 1 year

    # merge diagnoses from CPRD and HES, and map all codes to ICD10
    source_config, prep_config = params['raw_file_path'], params['mid_stage_file_path']

    diagnoses = read_parquet(spark.sqlContext, os.path.join(prep_config['mid_stage_dir'],
                                                            prep_config['mapped_diagnosis']))

    diagnoses_icd = diagnoses.filter(F.col('type') == 'icd').withColumn('first', F.col('code').substr(0, 1)).filter(
        F.col('first').isin(*['Z', 'V', 'R', 'U', 'X', 'Y']) == False) \
        .select(['patid', 'eventdate', 'code', 'type', 'source'])

    diagnoses_read = diagnoses.filter(F.col('type') == 'read') \
        .withColumn('first', F.col('code').substr(0, 1)) \
        .filter(F.col('first').isin(*['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Z', 'U']) == False) \
        .select(['patid', 'eventdate', 'code', 'type', 'source'])

    diagnoses = diagnoses_icd.union(diagnoses_read)

    # filter by time
    diagnoses = check_time(diagnoses, 'eventdate', time_a=global_params['start'], time_b=global_params['end'])

    # filter by patient
    rep_pat = read_parquet(spark.sqlContext, os.path.join(prep_config['mid_stage_dir'], prep_config['downstream']))
    rep_pat = rename_col(rep_pat, 'patid', 'patid_eligible')
    diagnoses = diagnoses.join(rep_pat, diagnoses.patid == rep_pat.patid_eligible, 'left')
    diagnoses = diagnoses.filter(F.col('patid_eligible').isNotNull()).drop('patid_eligible')


    #format diagnoses into bert format
    demographic = Patient(read_txt(spark.sc, spark.sqlContext, path=source_config[
        'patient'])).accept_flag().yob_calibration().cvt_crd2date().cvt_tod2date().cvt_deathdate2date().get_pracid().drop(
        'accept') \
        .select(['patid', 'yob'])
    diagnoses = diagnoses.join(demographic, diagnoses.patid == demographic.patid, 'inner').drop(demographic.patid)
    diagnoses = EHR(diagnoses).cal_age('eventdate', 'yob', year=False).select(
        ['patid', 'eventdate', 'age', 'code', 'yob']).dropDuplicates()

    # group by date
    diagnoses = diagnoses.groupby(['patid', 'eventdate']).agg(F.collect_list('code').alias('code'),
                                                              F.collect_list('age').alias('age'),
                                                              F.first('yob').alias('yob'))

    diagnoses = EHR(diagnoses).array_add_element('code', 'SEP')

    # add extra age to fill the gap of sep
    extract_age = F.udf(lambda x: x[0])
    diagnoses = diagnoses.withColumn('age_temp', extract_age('age')).withColumn('age', F.concat(F.col('age'), F.array(
        F.col('age_temp')))).drop('age_temp')

    w = Window.partitionBy('patid').orderBy('eventdate')
    # sort and merge ccs and age
    diagnoses = diagnoses.withColumn('code', F.collect_list('code').over(w)) \
        .withColumn('age', F.collect_list('age').over(w)) \
        .withColumn('eventdate', F.collect_list('eventdate').over(w)) \
        .groupBy('patid').agg(F.max('code').alias('code'), F.max('age').alias('age'),
                              F.max('eventdate').alias('eventdate'))

    diagnoses = EHR(diagnoses).array_flatten('code').array_flatten('age')

    ## HF data generation list
    # data = read_parquet(spark.sqlContext, '/home/shared/yikuan/Multimortality/data/temp.parquet')
    data = diagnoses
    leng = F.udf(lambda s: len([i for i in range(len(s)) if s[i] == 'SEP']))
    lengevent = F.udf(lambda s: len(s))
    span = F.udf(lambda x: int(x[-1]) - int(x[0]))
    # cvt2sting=  F.udf(lambda x: [str(each) for each in x])

    data = data.withColumn('length', leng(global_params['code_col'])).withColumn('length_ev', lengevent('eventdate'))
    # data = data[data['length'] >= global_params['min_visit']]
    data = data.withColumn('span', span('age'))
    # data = data[data['span'] > global_params['span']]
    # data = data.withColumn('eventdate', cvt2sting('eventdate'))

    schema = StructType([StructField('patid', StringType(), True),
                         StructField('firstDate', DateType(), True),
                         StructField('lastDate', DateType(), True),
                         StructField('label', IntegerType(), True),
                         StructField('HFDate', DateType(), True)
                         ])

    def full_gen(code, age, patid, eventdate):
        hf_index = [i for i in range(len(code)) if code[i] in global_params['hf_list']]
        sep_index = [i for i in range(len(code)) if code[i] == 'SEP']
        if len(hf_index) == 0:
            if len(sep_index) <= global_params['min_visit']:
                sample_index = None
            else:
                last_usable = [i for i in range(len(sep_index)) if
                               (int(age[sep_index[-1]]) - int(age[sep_index[i]])) >= global_params['span']]
                if len(last_usable) == 0:
                    sample_index = None
                else:
                    last_usable = last_usable[-1]
                    if (last_usable + 1) <= (global_params['min_visit'] - 1):
                        sample_index = None
                    else:
                        sample_index = random.choice(np.arange(global_params['min_visit'] - 1, last_usable + 1))
            if sample_index is None:
                return '0', datetime.datetime(1000, 1, 1), datetime.datetime(1000, 1, 1), -1, datetime.datetime(1000, 1,
                                                                                                                1)
            else:
                patid = patid
                label = 0
                firstDate = eventdate[0]
                lastDate = eventdate[sample_index]
                return patid, firstDate, lastDate, label, datetime.datetime(1000, 1, 1)
        else:
            hf_index = hf_index[0]
            if hf_index < sep_index[0]:
                return '0', datetime.datetime(1000, 1, 1), datetime.datetime(1000, 1, 1), -1, datetime.datetime(1000, 1,
                                                                                                                1)
            else:
                visit_index = \
                    [i for i in range(1, len(sep_index)) if sep_index[i] > hf_index and sep_index[i - 1] < hf_index][0]
                if (int(age[sep_index[visit_index]]) - int(age[sep_index[visit_index - 1]])) < global_params['span']:
                    sample_index = visit_index - 1
                    patid = patid
                    firstDate = eventdate[0]
                    lastDate = eventdate[sample_index]
                    hfDate = eventdate[visit_index]
                    label = 1
                    return patid, firstDate, lastDate, label, hfDate
                else:
                    return '0', datetime.datetime(1000, 1, 1), datetime.datetime(1000, 1, 1), -1, datetime.datetime(
                        1000, 1, 1)

    test_udf = F.udf(full_gen, schema)
    data = data.select(test_udf(global_params['code_col'], 'age', 'patid', 'eventdate').alias("test"))
    data = data.select("test.*")

    data = data[data['patid'] != '0']
    data = data[data['label'] != -1]
    data = data[data['firstDate'] != datetime.datetime(1000, 1, 1)]
    data = data[data['lastDate'] != datetime.datetime(1000, 1, 1)]

    data_non = data.filter(F.col('label') == 0)
    data_HF = data.filter(F.col('label') == 1)

    cal_HF_duration = (F.unix_timestamp('HFDate', "yyyy-MM-dd") - F.unix_timestamp('lastDate', "yyyy-MM-dd"))
    data_HF = data_HF.withColumn('duration', cal_HF_duration).withColumn('duration',
                                                                         (F.col('duration') / 3600 / 24).cast(
                                                                             'integer'))
    data_HF_eligible = data_HF.filter(F.col('duration') >= low_bound) \
        .select(['patid', 'firstDate', 'lastDate', 'label', 'HFDate'])

    data_HF_modif = data_HF.filter(F.col('duration') < low_bound) \
        .withColumn('high_bound', F.lit(high_bound)).withColumn('low_bound', F.lit(low_bound)) \
        .withColumn('last_low', F.expr("date_sub(HFDate,high_bound)")) \
        .withColumn('last_high', F.expr("date_sub(HFDate,low_bound)"))

    data = read_parquet(spark.sqlContext, os.path.join(prep_config['mid_stage_dir'],prep_config['mapped_diagnosis']))

    data = data.join(data_HF_modif, data.patid == data_HF_modif.patid, 'inner').drop(data.patid) \
        .where((F.col('eventdate') > F.col('last_low')) & (F.col('eventdate') < F.col('last_high'))) \
        .orderBy(F.rand()) \
        .groupby(['patid', 'firstDate', 'lastDate', 'label', 'HFDate']) \
        .agg(F.first('eventdate').alias('eventdate')).withColumn('lastDate', F.col('eventdate')) \
        .select(['patid', 'firstDate', 'lastDate', 'label', 'HFDate'])

    data = data.union(data_HF_eligible).union(data_non)

    # remove none and duplicate
    output = params['mid_stage_file_path']
    check_dir(output['mid_stage_dir'])

    data.write.parquet(os.path.join(output['mid_stage_dir'], output['hf_data']))