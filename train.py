"""Training script. Runs training/evaluation loops for resnet18 model."""
from argparse import ArgumentParser
from typing import Tuple
import sys
sys.path.append('.')
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from modules.dataset import VisibilityDataset
from modules.model import ResnetLightning
from modules.transforms import TransformationsBase
from modules.utils import random_split_train_val_test_stratified
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import WeightedRandomSampler

task_params = {
    'gpu_idx': 2,
    'num_sanity_val_steps': 0,
    'max_epochs': 150,
    'num_resnet_layers': 18,
    'out_features': 3,  # we have 3 outputs
    'num_workers': 2,
    'batch_size': 64,
    'learning_rate': 0.0001,
    'early_stop_monitor': 'val_loss',
    'early_stop_mode': 'min',
    'early_stop_patience': 30,  #epochs
    'save_top_k': 1,
    'tags': ['resnet18']
}

# we are using the pixel mean std of ImageNet training set
# because our pretrained model used these values for image normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def prepare_datasets(
    images_root_dir: str,
    data_csv_path: str,
    transform: TransformationsBase,
) -> Tuple[VisibilityDataset, VisibilityDataset, VisibilityDataset]:
    """Splits the input data labels into train/val/test components and constructs pytorch datasets.

    Args:
        images_root_dir (str): path to the root directory of video frames.
        data_csv_path (str): path to the data csv file having image file names and labels.
        transform (TransformationsBase): image pixel transforms for dataset construction.

    Returns:
        Tuple[VisibilityDataset, VisibilityDataset, VisibilityDataset]: 
            Tuple of training, validation, testing datasets.
    """
    df = pd.read_csv(data_csv_path)

    df = df.rename(columns={' LABEL': 'LABEL'})

    df['LABEL'] = df['LABEL'].apply(lambda x: x.strip())

    df_train, df_val, df_test = random_split_train_val_test_stratified(df, label_colname='LABEL')

    train_dataset = VisibilityDataset(images_root_dir=images_root_dir,
                                      dataframe=df_train,
                                      spatial_transform=transform.train_transform,
                                      norm_transform=transform.norm_transform)

    valid_dataset = VisibilityDataset(images_root_dir=images_root_dir,
                                      dataframe=df_val,
                                      spatial_transform=transform.eval_transform,
                                      norm_transform=transform.norm_transform)

    test_dataset = VisibilityDataset(images_root_dir=images_root_dir,
                                     dataframe=df_test,
                                     spatial_transform=transform.eval_transform,
                                     norm_transform=transform.norm_transform)

    return train_dataset, valid_dataset, test_dataset


def get_balanced_weights_per_source(df: pd.DataFrame) -> torch.tensor:
    """Computes weights per data sample based on the data source.
    Sample coming from less prevalent source receives higher weight.

    Args:
        df (pd.DataFrame): input dataframe of video frames and source.

    Returns:
        torch.tensor: tensor of sample weights.
    """
    source_weight = 1. / df.LABEL.value_counts(normalize=True)
    sample_weight = df.LABEL.map(source_weight)
    return torch.from_numpy(sample_weight.values)


def prepare_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers):
    samples_weight = get_balanced_weights_per_source(train_dataset.dataframe)

    train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_dl = data.DataLoader(train_dataset,
                               batch_size=batch_size,
                               sampler=train_sampler,
                               num_workers=num_workers,
                               pin_memory=True)
    valid_dl = data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_dl = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_dl, valid_dl, test_dl


def prepare_trainer_callbacks(early_stop_monitor, early_stop_mode, early_stop_patience, ckpt_dirpath, ckpt_save_top_k):
    early_stop_callback = pl.callbacks.EarlyStopping(monitor=early_stop_monitor,
                                                     mode=early_stop_mode,
                                                     patience=early_stop_patience)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename=f'{{epoch:02d}}-{{{early_stop_monitor}:.2f}}',
                                                       dirpath=ckpt_dirpath,
                                                       monitor=early_stop_monitor,
                                                       mode=early_stop_mode,
                                                       save_top_k=ckpt_save_top_k)

    return checkpoint_callback, early_stop_callback


def main(data_csv_path: str, images_root_dir: str, artifacts_dir: str):
    """Main function which starts training."""

    transform = TransformationsBase(scale_size=224, norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD)

    train_dataset, valid_dataset, test_dataset = prepare_datasets(images_root_dir, data_csv_path, transform)

    train_dl, valid_dl, test_dl = prepare_dataloaders(train_dataset, valid_dataset, test_dataset,
                                                      task_params['batch_size'], task_params['num_workers'])

    module = ResnetLightning(
        out_features=task_params['out_features'],
        learning_rate=task_params['learning_rate']
    )

    trainer_callbacks = prepare_trainer_callbacks(task_params['early_stop_monitor'], task_params['early_stop_mode'],
                                                  task_params['early_stop_patience'], artifacts_dir,
                                                  task_params['save_top_k'])

    logger = TensorBoardLogger('tb_logs', name='my_visibility_model')
    trainer = pl.Trainer(max_epochs=task_params['max_epochs'],
                         num_sanity_val_steps=0,
                         callbacks=list(trainer_callbacks),
                         gpus=[task_params['gpu_idx']],
                         logger=[logger])

    trainer.fit(module, train_dataloader=train_dl, val_dataloaders=valid_dl)

    trainer.test(test_dataloaders=test_dl)


if __name__ == '__main__':
    parser = ArgumentParser(
        description=
        "This script runs training and evaluation of the Resnet18 model for ID document visibility classification.")
    parser.add_argument('--data_csv_path', type=str, help="path to the csv file having image file names and labels.", default='data/gicsd_labels.csv')
    parser.add_argument('--images_root_dir', type=str, help="Root directory path to images folder.", default='data/images_processed')
    parser.add_argument('--artifacts_dir',
                        type=str,
                        help="The root artifacts directory which will hold model artifacts.", default='artifacts/')

    args = parser.parse_args()
    main(**vars(args))
