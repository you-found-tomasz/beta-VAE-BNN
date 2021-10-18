from vae.vae import VaeModel
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import datetime
from vae.config_loader import load_config
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from pytorch_lightning.utilities.cloud_io import load as pl_load
import os
import torchvision.transforms as transforms
from vae.utils import parser
import math


def load_vae_model():
    args = parser()

    if args.lbb == True:
        gpu_number = args.gpu_number  # Adapt: This is the GPU you are using. Possible values: 0 - 7
        username = "tomasz"  # Adapt: Change this string to your username

        from os import environ as cuda_environment
        import setproctitle

        setproctitle.setproctitle(username + str(" python 3.6"))
        cuda_environment["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

    t_raw = datetime.datetime.now()
    time_stamp = str(t_raw.strftime("%y") + "_" + t_raw.strftime("%m") + "_" + t_raw.strftime("%d") + "_" + str(
        datetime.datetime.now().time())[:5])
    tb_logger = pl_loggers.TensorBoardLogger('logs/', name="{}_{}".format(time_stamp, args.logging_string))

    config_file_name = args.network_config
    full_config = load_config(network_enc=args.network_enc, network_dec=args.network_dec, raytune=False,
                              config_file_name=config_file_name)

    vae_model = LitDecoderModel(config=full_config, args=args)
    trainer = pl.Trainer(logger=tb_logger, gpus=args.gpu, max_epochs=args.max_epochs, stochastic_weight_avg=True,
                         accumulate_grad_batches={4: 1, 8: 2, 12: 3, 15: 4},
                         gradient_clip_val=0.0001)  # , overfit_batches=1)#, fast_dev_run=True)

    return trainer, vae_model, args, tb_logger


def single_inference(x):
    _, vae_model, _, _ = load_vae_model()
    trained_model = vae_model.load_from_checkpoint(checkpoint_path="trained_model/trained_model.ckpt")
    trained_model.eval()

    train_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    x = train_transform(x)
    x_short = x[:, :, :4000]
    x_short = x_short.reshape(x.shape[0], -1)
    z, _ = trained_model.encoder(x_short)

    return z


def train_mnist_tune(config, num_epochs=10, num_gpus=0, args=[]):
    model = LitDecoderModel(config, args)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=math.ceil(num_gpus),  # If fractional GPUs passed in, convert to int.
        logger=pl_loggers.TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "ptl/elbo_loss",
                    "number_params": "ptl/number_params",
                },
                on="epoch_end")
        ])
    trainer.fit(model)


def train_mnist_tune_checkpoint(config, checkpoint_dir=None, num_epochs=10, num_gpus=0, args=[]):
    print(tune.get_trial_dir())
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accumulate_grad_batches={1: 2, 2: 4},
        gradient_clip_val=0.0001,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=pl_loggers.TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "ptl/elbo_loss",
                    "number_params": "ptl/number_params",
                },
                filename="checkpoint",
                on="epoch_end")
        ])
    if checkpoint_dir:
        # Currently, this leads to errors:
        # model = LightningMNISTClassifier.load_from_checkpoint(
        #     os.path.join(checkpoint, "checkpoint"))
        # Workaround:
        ckpt = pl_load(
            os.path.join(checkpoint_dir, "checkpoint"),
            map_location=lambda storage, loc: storage)
        model = LitDecoderModel._load_model_state(ckpt, config=config, args=args)
        trainer.current_epoch = ckpt["epoch"]
    else:
        model = LitDecoderModel(config=config, args=args)
        
    trainer.fit(model)


def tune_mnist_asha(num_samples=10, num_epochs=10, gpus_per_trial=0, args=[]):
    full_config = load_config(args.network_enc, args.network_dec, args.raytune, args.network_config)
    param_columns = list(full_config)

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=2,  # original 1
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=param_columns,
        metric_columns=["loss", "number_params", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_mnist_tune,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
            args=args,
        ),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=full_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose=1,
        local_dir="./results",
        name="tune_mnist_asha")

    print("Best hyperparameters found were: ", analysis.best_config)


def tune_mnist_pbt(num_samples=10, num_epochs=10, gpus_per_trial=0, args=[]):
    full_config = load_config(args.network_enc, args.network_dec, args.raytune, args.network_config)
    param_columns = list(full_config)[:3]
    full_config["lr"] = tune.loguniform(1e-5, 2e-5)
    full_config["batch_size"] = tune.choice([4])

    scheduler = PopulationBasedTraining(
        # perturbation_interval=25,
        perturbation_interval=10,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-13, 1e-4),
            "batch_size": [8, 16, 64, 128, 256, 512, 1024]
        })

    reporter = CLIReporter(
        parameter_columns=param_columns,
        metric_columns=["loss", "val_loss", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_mnist_tune_checkpoint,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
            args=args, ),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=full_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="/scratch/zaluskat/Semester_project/tomasz_zaluska/210901/results",
        name="tune_mnist_pbt")

    print("Best hyperparameters found were: ", analysis.best_config)
