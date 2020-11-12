from utils.yaml_act import yaml_load
from utils.arg_parse import arg_paser
from CPRD.spark import spark_init
from data_preprocessing.task import *


def main():
    args = arg_paser()
    params = yaml_load(args.params)
    params.update(args.update_params)
    print(args)

    spark = spark_init()
    run = eval(params['task'])
    run(params, spark)


if __name__ == "__main__":
    main()