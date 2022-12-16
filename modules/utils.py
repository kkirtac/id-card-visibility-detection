import math
import os
import random
from typing import Tuple

import cv2
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def sample_load_images(image_path_list, num_sample_images=16):
    """Samples num_sample_images image path from the list image_path_list.
    Next, every image in the path is loaded and returned in a list.

    Args:
        image_path_list (list): list of absolute path to all images.
        num_sample_images (int, optional): number of images to sample from image_path_list. Defaults to 16.
    """
    sampled_images = random.sample(image_path_list, num_sample_images)

    return [cv2.imread(img) for img in sampled_images]


def show_images(img_list, figsize=(10, 10)):
    """Show the given list of images on a grid-like view.

    Args:
        img_list (list): list of image objects (numpy array).
        figsize (float, float): width, height in inches.
    """
    numcols = 4
    numrows = len(img_list) // numcols
    if numrows < 1:
        numrows = 1
        numcols = len(img_list)

    _, axs = plt.subplots(numrows, numcols, figsize=figsize)
    axs = axs.flatten()
    for img, ax in zip(img_list[:len(axs)], axs):
        ax.imshow(img)
    plt.show()


def get_color_histogram(img):
    """Compute the per-channel histogram of a color image and return in a list.

    Args:
        img (numpy.array): a 3-channel BGR image.

    Returns:
        list: a list of histogram per image channel.
    """
    hist = []

    color = ('b', 'g', 'r')

    for i, _ in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist.append(histr)
    return hist


def plot_color_histogram(img_list):
    """Plots the color histogram of a list of BGR images.

    Args:
        img_list (list): list of 3-channel input BGR images (numpy array).
    """
    grid_size = math.floor(math.sqrt(len(img_list)))

    _, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axs = axs.flatten()

    color = ('b', 'g', 'r')

    for img, ax in zip(img_list[:len(axs)], axs):
        hist = get_color_histogram(img)
        for i, col in enumerate(color):
            ax.plot(hist[i], color=col)
            ax.set_xlim(0, 256)
    plt.show()


def save_images(img_list, src_img_names, dst_root_dir):
    """Save the given list of images into a directory.

    Args:
        img_list (list): list of input images.
        src_img_names (list): list of image names.
        dst_root_dir (str): destination directory.
    """
    os.makedirs(dst_root_dir, exist_ok=True)
    for img, src_img_name in zip(img_list, src_img_names):
        dst_path = os.path.join(dst_root_dir, src_img_name)
        cv2.imwrite(dst_path, img)


def random_split_train_val_test_stratified(df_images_labels: pd.DataFrame, 
                                    label_colname:str,
                                    train_pct: float = 0.7,
                                    val_pct: float = 0.1,
                                    test_pct: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Performs stratified sampling according to the visibility labels.

    Args:
        df_images_labels (pd.DataFrame): Input dataframe of images and visibility labels.
        label_colname (str): Labels column name.
        train_pct (float, optional): Training set percentage. Defaults to 0.7.
        val_pct (float, optional): Validation set percentage. Defaults to 0.1.
        test_pct (float, optional): Testing set percentage. Defaults to 0.2.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A dataframe for each of training, validation and
            testing partitions.
    """
    df_train, df_test = train_test_split(df_images_labels, test_size=test_pct, random_state=10, stratify=df_images_labels[label_colname])

    df_train, df_val = train_test_split(df_train, test_size=val_pct / (val_pct + train_pct), random_state=10, stratify=df_train[label_colname])

    return df_train.reset_index().drop('index',
                                       axis=1), df_val.reset_index().drop('index',
                                                                          axis=1), df_test.reset_index().drop('index',
                                                                                                              axis=1)
