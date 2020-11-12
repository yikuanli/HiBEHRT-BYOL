import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame
import pyspark
from CPRD.utils import cvt_str2time
# def rename_col(df, old, new):
#     """rename pyspark dataframe column"""
#     return df.withColumnRenamed(old, new)


class Patient(DataFrame):
    def __init__(self, df):
        super(self.__class__, self).__init__(df._jdf, df.sql_ctx)

    def accept_flag(self):
        """select rows with accpt = 1"""
        return Patient(self.where(self.accept == 1))

    def yob_calibration(self):
        """decode yob to regular year"""
        return Patient(self.withColumn('yob', sum([self.yob, 1800]).cast(pyspark.sql.types.IntegerType())))

    def cvt_tod2date(self):
        """convert tod from string to date"""
        return Patient(self.withColumn('tod', cvt_str2time(self, 'tod')))

    def cvt_deathdate2date(self):
        """convert deathdate from string to date"""
        return Patient(self.withColumn('deathdate', cvt_str2time(self, 'deathdate')))

    def cvt_crd2date(self):
        """convert crd from string to date"""
        return Patient(self.withColumn('crd', cvt_str2time(self, 'crd')))

    def get_pracid(self):
        """get pracid from patid inorder to join with practice table"""
        return Patient(self.withColumn('pracid', self['patid'].substr(-3, 3).cast(pyspark.sql.types.IntegerType())))


class Clinical(DataFrame):
    def __init__(self, df):
        super(self.__class__, self).__init__(df._jdf, df.sql_ctx)

    def cvtEventDate2Time(self):
        """ convert eventdate from strnig to date type"""
        return Clinical(self.withColumn('eventdate', cvt_str2time(self, 'eventdate')))

    def rm_eventdate_medcode_empty(self):
        """rm row with empty eventdate or medcode"""
        return Clinical(self.filter((F.col('eventdate') != '') & (F.col('medcode') != '')))


class Practice(DataFrame):
    def __init__(self, df):
        super(self.__class__, self).__init__(df._jdf, df.sql_ctx)

    def cvt_lcd2date(self):
        """convert lcd from string to date"""
        return Practice(self.withColumn('lcd', cvt_str2time(self, 'lcd')))

    def cvt_uts2date(self):
        """convert uts from string to date"""
        return Practice(self.withColumn('uts', cvt_str2time(self, 'uts')))


class Diagnosis(DataFrame):
    def __init__(self, df):
        super(self.__class__, self).__init__(df._jdf, df.sql_ctx)

    def cvt_admidate2date(self):
        """conver admidate from string to date"""
        df = self.withColumn('admidate', F.concat(F.col('admidate').substr(7, 4), F.col('admidate').substr(4,2), F.col('admidate').substr(1, 2)))
        return Diagnosis(df.withColumn('admidate', cvt_str2time(df, 'admidate')))

    def icd_rm_dot(self):
        """remove '.' from ICD code"""
        replace = F.udf(lambda x: x.replace('.', ''))
        return Diagnosis(self.withColumn('ICD', replace('ICD')))

    def rm_date_icd_empty(self):
        """remove admidate or icd code is empty"""
        return Diagnosis(self.filter((F.col('admidate') != '') & (F.col('ICD') != '')))


class Hes(DataFrame):
    def __init__(self, df):
        super(self.__class__, self).__init__(df._jdf, df.sql_ctx)

    def cvt_string2date(self, col):
        df = self.withColumn(col, F.concat(F.col(col).substr(7, 4), F.col(col).substr(4, 2), F.col(col).substr(1, 2)))
        return Hes(df.withColumn(col, cvt_str2time(df, col)))


class Therapy(DataFrame):
    def __init__(self, df):
        super(self.__class__, self).__init__(df._jdf, df.sql_ctx)

    def rm_eventdate_prodcode_empty(self):
        """rm row with empty eventdate or medcode"""
        return Therapy(self.filter((F.col('eventdate') != '') & (F.col('prodcode') != '')))

    def cvtEventDate2Time(self):
        """ convert eventdate from strnig to date type"""
        return Therapy(self.withColumn('eventdate', cvt_str2time(self, 'eventdate')))


class EHR(DataFrame):
    def __init__(self, df):
        super(self.__class__, self).__init__(df._jdf, df.sql_ctx)

    def cal_age(self, event_date, yob, year=True, name='age'):
        if year:
            age_cal = F.udf(lambda x, y : x.year - y)
        else:
            # assume people born in January
            age_cal = F.udf(lambda x, y : (x.year * 12 + x.month) - (y * 12 + 1))

        return EHR(self.withColumn(name, age_cal(F.col(event_date), F.col(yob))))

    def set_col_to_str(self, col):
        return EHR(self.withColumn(col, F.col(col).cast('string')))

    def array_add_element(self, col, element):
        return EHR(self.withColumn(col, F.concat(F.col(col), F.array(F.lit(element)))))

    def array_flatten(self, col):
        return EHR(self.withColumn(col, F.flatten(F.col(col))))