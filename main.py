import os
import funcy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ujson
import pandas as pd
import re

from functools import reduce
from fnmatch import fnmatch
from itertools import combinations
from PIL import Image
from scipy import stats
from scipy.stats import ks_2samp
from skimage.io import imread
from skimage.morphology import square, disk, erosion, dilation, opening, closing, convex_hull_image
from sklearn.metrics import cohen_kappa_score
from torchvision import transforms
from tqdm import tqdm

font = {"size": 22}
matplotlib.rc("font", **font)

selem = square(5)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ISIC_ARCHIVE_INPUTS_PATH = os.path.join(BASE_PATH, "data", "isic_archive", "inputs")
ISIC_ARCHIVE_TARGETS_PATH = os.path.join(BASE_PATH, "data", "isic_archive", "targets")
ISIC_2017_TARGETS_PATH = os.path.join(BASE_PATH, "data", "isic2017", "train", "targets")
ISIC_2018_TARGETS_PATH = os.path.join(BASE_PATH, "data", "isic2018", "train", "targets")
ISIC_ARCHIVE_KAPPA_SCORES_PATH = os.path.join("workshop_files", "isic_archive_kappa_scores.csv")
ISIC_2017_KAPPA_SCORES_PATH = os.path.join("workshop_files", "isic_2017_kappa_scores.csv")
ISIC_2018_KAPPA_SCORES_PATH = os.path.join("workshop_files", "isic_2018_kappa_scores.csv")


def bounding_box(img):
    min_x, max_x, min_y, max_y = np.inf, 0, np.inf, 0

    for i, row in enumerate(img):
        for j, val in enumerate(row):
            if val > 250:
                if i < min_y:
                    min_y = i
                if i > max_y:
                    max_y = i

                if j < min_x:
                    min_x = j
                if j > max_x:
                    max_x = j

    if min_x == np.inf:
        min_x = 0
    if min_y == np.inf:
        min_y = 0

    new_img = np.zeros_like(img)
    bounding_box = np.ones((max_y - min_y, max_x - min_x))
    new_img[min_y:max_y, min_x:max_x] = bounding_box

    return new_img


def load_target_image(fpath):
    img = imread(fpath, as_gray=True)
    return Image.fromarray(img)


def transform(transform_fn, pil_image, mult=1, **kwargs):
    img = np.array(pil_image)
    img = mult * transform_fn(img, **kwargs).astype(np.uint8)
    img = Image.fromarray(img)

    return img.point(lambda p: p > 255 // 2 and 255)
