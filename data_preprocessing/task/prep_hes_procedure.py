from CPRD.table import Patient,Practice,Clinical, Diagnosis, Therapy, Hes
from CPRD.spark import read_txt, read_csv
import pyspark.sql.functions as F
from CPRD.utils import rename_col, check_dir, check_time
import os


def main(params, spark):
    # merge diagnoses from CPRD and HES, and map all codes to ICD10
    config = params['raw_file_path']

    # read data and process each table
    hes_procedure = read_txt(spark.sc, spark.sqlContext, config['hes_procedure']).select(['patid', 'OPCS', 'evdate'])
    hes_procedure = rename_col(hes_procedure, 'evdate', 'eventdate')

    def cvt_str2timeProcedure(df, col):
        """convert column from string to date type"""
        return F.to_date(
            F.concat(df[col].substr(7, 4), F.lit('-'), df[col].substr(4, 2), F.lit('-'), df[col].substr(1, 2)))

    hes_procedure = hes_procedure.withColumn('eventdate', cvt_str2timeProcedure(hes_procedure, 'eventdate')).dropna()

    hes_procedure = check_time(hes_procedure, 'eventdate', time_a=1985, time_b=2015)

    # remove none and duplicate
    output = params['mid_stage_file_path']
    check_dir(output['mid_stage_dir'])

    hes_procedure.write.parquet(os.path.join(output['mid_stage_dir'], output['hes_procedure']))