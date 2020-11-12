import os
import pyspark.sql.functions as F


def rename_col(df, old, new):
    """rename pyspark dataframe column"""
    return df.withColumnRenamed(old, new)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_time(df, col, time_a=1985, time_b=2016):
    """keep data with date between a and b"""
    year = F.udf(lambda x: x.year)
    df = df.withColumn('Y', year(col))
    df = df.filter(F.col('Y') >= time_a)
    df = df.filter(F.col('Y') <= time_b).drop('Y')
    return df


def cvt_str2time(df, col):
    """convert column from string to date type"""
    return F.to_date(F.concat(df[col].substr(1, 4), F.lit('-'), df[col].substr(5, 2), F.lit('-'), df[col].substr(7, 2)))
