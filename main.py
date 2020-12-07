from utils.yaml_act import yaml_load, yaml_save
from utils.arg_parse import arg_paser
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloaders import *
from models import *
from utils.callback import CheckpointEveryNSteps
import torch
from pytorch_lightning.callbacks import LearningRateMonitor


def main():
    print('number of CUDA device available:', torch.cuda.device_count())

    args = arg_paser()

    # process config
    params = yaml_load(args.params)
    params.update(args.update_params)
    print(args)

    env_params, base_params, train_params, eval_params, callback_params = \
        params['env_params'], params['base_params'], \
        params['train_params'], params['eval_params'], params['callback_params']

    # set up logging and save updated config file
    save_path = args.params if args.save_path is None else args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    yaml_save(params, save_path + '/config.yaml')

    # define logger
    logger = TensorBoardLogger(save_path, name='my_log')

    print('initialize data loader')
    # create data loader
    input_fn = eval(base_params['dataloader'])
    train_params.update(base_params['dataloader_params'])
    trainloader = input_fn(
        params=train_params
    )

    eval_params.update(base_params['dataloader_params'])
    evalloader = input_fn(
        params=eval_params
    )

    print('initialize model')
    # create the model and optimiser
    model = eval(base_params['model'])
    model_params = base_params['model_params']
    model_params.update(base_params['dataloader_params'])
    model = model(model_params)

    # set up training environment
    env_params.update({'logger': logger})
    env_params.update({'default_root_dir': os.path.join(save_path, 'checkpoint')})

    callback_params.update({'dirpath': save_path})
    checkpoint_callback = ModelCheckpoint(**callback_params)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [CheckpointEveryNSteps(20), lr_monitor]
    env_params.update({'checkpoint_callback': checkpoint_callback, 'callbacks': callbacks})
    trainer = pl.Trainer(**env_params)

    # train and evaluate model
    trainer.fit(model, trainloader, evalloader)


if __name__ == "__main__":
    main()