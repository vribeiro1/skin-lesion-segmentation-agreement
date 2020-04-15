import os
import funcy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import re

from fnmatch import fnmatch
from itertools import combinations
from PIL import Image
from scipy.stats import ks_2samp
from skimage.io import imread
from skimage.morphology import square, opening, closing, convex_hull_image
from sklearn.metrics import cohen_kappa_score
from torchvision import transforms
from tqdm import tqdm

font = {"size": 22}
matplotlib.rc("font", **font)

selem = square(5)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
WORKSHOP_FILES_PATH = os.path.join(BASE_PATH, "workshop_files")

ISIC_ARCHIVE_INPUTS_PATH = os.path.join(BASE_PATH, "data", "isic_archive", "inputs")
ISIC_ARCHIVE_TARGETS_PATH = os.path.join(BASE_PATH, "data", "isic_archive", "targets")
ISIC_2017_TARGETS_PATH = os.path.join(BASE_PATH, "data", "isic2017", "train", "targets")
ISIC_2018_TARGETS_PATH = os.path.join(BASE_PATH, "data", "isic2018", "train", "targets")
ISIC_ARCHIVE_KAPPA_SCORES_PATH = os.path.join(WORKSHOP_FILES_PATH, "isic_archive_kappa_scores.csv")
ISIC_2017_KAPPA_SCORES_PATH = os.path.join(WORKSHOP_FILES_PATH, "isic_2017_kappa_scores.csv")
ISIC_2018_KAPPA_SCORES_PATH = os.path.join(WORKSHOP_FILES_PATH, "isic_2018_kappa_scores.csv")


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

########################################################################################################################
#
# Get the file IDs for each dataset
#
########################################################################################################################


full_isic_archive_ids = funcy.walk(lambda fname: fname.split(".")[0], os.listdir(ISIC_ARCHIVE_INPUTS_PATH))

isic_archive_ids = list(
    sorted(
        set(
            funcy.walk(
                lambda fname: re.match(r"(ISIC_[0-9]+)_segmentation_[0-9]+.png", fname).group(1),
                filter(lambda fname: fname.endswith(".png"), os.listdir(ISIC_ARCHIVE_TARGETS_PATH))
            )
        )
    )
)

isic_2017_ids = list(
    sorted(
        set(
            funcy.walk(
                lambda fname: re.match(r"(ISIC_[0-9]+)_segmentation.png", fname).group(1),
                filter(lambda fname: fname.endswith(".png"), os.listdir(ISIC_2017_TARGETS_PATH))
            )
        )
    )
)

isic_2018_ids = list(
    sorted(
        set(
            funcy.walk(
                lambda fname: re.match(r"(ISIC_[0-9]+)_segmentation.png", fname).group(1),
                filter(lambda fname: fname.endswith(".png"), os.listdir(ISIC_2018_TARGETS_PATH))
            )
        )
    )
)

total_ids = list(
    sorted(
        set(isic_archive_ids + isic_2017_ids + isic_2018_ids)
    )
)

print(f"""
Full ISIC Archive size: {len(full_isic_archive_ids)}

Number of images with segmentation masks:

ISIC Archive: {len(isic_archive_ids)}
ISIC 2017: {len(isic_2017_ids)}
ISIC 2018: {len(isic_2018_ids)}

Total unique images: {len(total_ids)}
""")

########################################################################################################################
#
# Get files that have more than one mask and build data frame for holding distributions
#
########################################################################################################################


input_img_list = list(filter(lambda x: x.endswith("jpg"), os.listdir(ISIC_ARCHIVE_INPUTS_PATH)))
target_img_list = [
    (fname + "." + ext, list(filter(lambda x: x.startswith(fname), os.listdir(ISIC_ARCHIVE_TARGETS_PATH))))
    for fname, ext in map(lambda x: x.split("."), input_img_list)
]
non_unique_masks_list = list(filter(lambda x: len(x[1]) > 1, target_img_list))

data = []
for isic_id in isic_archive_ids:
    masks = list(filter(lambda fname: fnmatch(fname, f"{isic_id}_segmentation_*"),
                        os.listdir(ISIC_ARCHIVE_TARGETS_PATH)))
    n_masks = max(len(masks), 1)

    data.append((isic_id, masks, len(masks)))

df_n_masks_isic_archive = pd.DataFrame(data=data, columns=["id", "masks", "n_masks"])
df_n_masks_isic_2017 = df_n_masks_isic_archive[df_n_masks_isic_archive.id.isin(isic_2017_ids)]
df_n_masks_isic_2018 = df_n_masks_isic_archive[df_n_masks_isic_archive.id.isin(isic_2018_ids)]

df_n_masks_isic_archive.to_csv(os.path.join(WORKSHOP_FILES_PATH, "isic_archive_n_masks.csv"), index=False)
df_n_masks_isic_2017.to_csv(os.path.join(WORKSHOP_FILES_PATH, "isic_2017_n_masks.csv"), index=False)
df_n_masks_isic_2018.to_csv(os.path.join(WORKSHOP_FILES_PATH, "isic_2018_n_masks.csv"), index=False)

plt.figure(1, figsize=(20, 10))
sns.distplot(df_n_masks_isic_archive.n_masks, kde=False)
plt.title("ISIC Archive - Annotations per image lesion distribution")
plt.xlabel("Number of available masks")
plt.savefig(os.path.join(WORKSHOP_FILES_PATH, "isic_archive_n_masks_distr.pdf"))

plt.figure(2, figsize=(20, 10))
sns.distplot(df_n_masks_isic_2017.n_masks, kde=False)
plt.title("ISIC 2017 - Annotations per image lesion distribution")
plt.xlabel("Number of available masks")
plt.savefig(os.path.join(WORKSHOP_FILES_PATH, "isic_2017_n_masks_distr.pdf"))

plt.figure(3, figsize=(20, 10))
sns.distplot(df_n_masks_isic_2018.n_masks, kde=False)
plt.title("ISIC 2018 - Annotations per image lesion distribution")
plt.xlabel("Number of available masks")
plt.savefig(os.path.join(WORKSHOP_FILES_PATH, "isic_2018_n_masks_distr.pdf"))

########################################################################################################################
#
# Apply transformations to masks and calculate the cohen's kappa score
#
########################################################################################################################

cohen_kappa_mean_scores = []
opening_cohen_kappa_mean_scores = []
closing_cohen_kappa_mean_scores = []
convex_hull_cohen_kappa_mean_scores = []
opening_convex_hull_cohen_kappa_mean_scores = []
closing_convex_hull_cohen_kappa_mean_scores = []
bounding_box_cohen_kappa_mean_scores = []

########################################################################################################################
# No transformation
########################################################################################################################

for i, (img_fname, masks_fnames) in tqdm(enumerate(non_unique_masks_list)):
    masks_fpaths = funcy.walk(lambda x: os.path.join(ISIC_ARCHIVE_TARGETS_PATH, x), masks_fnames)
    masks = funcy.walk(lambda x: load_target_image(x), masks_fpaths)
    resized_masks = funcy.walk(lambda img: transforms.Resize(size=(int(256 * img.size[1] / img.size[0]), 256))(img),
                               masks)
    masks_arrays = funcy.walk(lambda x: np.array(x), resized_masks)
    flattened_masks_arrays = funcy.walk(lambda x: x.flatten(), masks_arrays)

    two_by_two_flattened_masks_arrays = list(combinations(flattened_masks_arrays, 2))
    scores = funcy.walk(lambda t: cohen_kappa_score(*t), two_by_two_flattened_masks_arrays)
    mean_cohen_kappa = np.mean(scores)
    cohen_kappa_mean_scores.append(mean_cohen_kappa)

########################################################################################################################
# Opening
########################################################################################################################

for i, (img_fname, masks_fnames) in tqdm(enumerate(non_unique_masks_list)):
    masks_fpaths = funcy.walk(lambda x: os.path.join(ISIC_ARCHIVE_TARGETS_PATH, x), masks_fnames)
    masks = funcy.walk(lambda x: load_target_image(x), masks_fpaths)
    resized_masks = funcy.walk(lambda img: transforms.Resize(size=(int(256 * img.size[1] / img.size[0]), 256))(img),
                               masks)
    transformed_masks = funcy.walk(lambda img: transform(opening, img, selem=selem), resized_masks)
    masks_arrays = funcy.walk(lambda x: np.array(x), transformed_masks)
    flattened_masks_arrays = funcy.walk(lambda x: x.flatten(), masks_arrays)

    two_by_two_flattened_masks_arrays = list(combinations(flattened_masks_arrays, 2))
    scores = funcy.walk(lambda t: cohen_kappa_score(*t), two_by_two_flattened_masks_arrays)
    mean_cohen_kappa = np.mean(scores)
    opening_cohen_kappa_mean_scores.append(mean_cohen_kappa)

########################################################################################################################
# Closing
########################################################################################################################

for i, (img_fname, masks_fnames) in tqdm(enumerate(non_unique_masks_list)):
    masks_fpaths = funcy.walk(lambda x: os.path.join(ISIC_ARCHIVE_TARGETS_PATH, x), masks_fnames)
    masks = funcy.walk(lambda x: load_target_image(x), masks_fpaths)
    resized_masks = funcy.walk(lambda img: transforms.Resize(size=(int(256 * img.size[1] / img.size[0]), 256))(img),
                               masks)
    transformed_masks = funcy.walk(lambda img: transform(closing, img, selem=selem), resized_masks)
    masks_arrays = funcy.walk(lambda x: np.array(x), transformed_masks)
    flattened_masks_arrays = funcy.walk(lambda x: x.flatten(), masks_arrays)

    two_by_two_flattened_masks_arrays = list(combinations(flattened_masks_arrays, 2))
    scores = funcy.walk(lambda t: cohen_kappa_score(*t), two_by_two_flattened_masks_arrays)
    mean_cohen_kappa = np.mean(scores)
    closing_cohen_kappa_mean_scores.append(mean_cohen_kappa)

########################################################################################################################
# Convex Hull of original image
########################################################################################################################

for i, (img_fname, masks_fnames) in tqdm(enumerate(non_unique_masks_list)):
    masks_fpaths = funcy.walk(lambda x: os.path.join(ISIC_ARCHIVE_TARGETS_PATH, x), masks_fnames)
    masks = funcy.walk(lambda x: load_target_image(x), masks_fpaths)
    resized_masks = funcy.walk(lambda img: transforms.Resize(size=(int(256 * img.size[1] / img.size[0]), 256))(img),
                               masks)
    transformed_masks = funcy.walk(lambda img: transform(convex_hull_image, img, mult=255), resized_masks)
    masks_arrays = funcy.walk(lambda x: np.array(x), transformed_masks)
    flattened_masks_arrays = funcy.walk(lambda x: x.flatten(), masks_arrays)

    two_by_two_flattened_masks_arrays = list(combinations(flattened_masks_arrays, 2))
    scores = funcy.walk(lambda t: cohen_kappa_score(*t), two_by_two_flattened_masks_arrays)
    mean_cohen_kappa = np.mean(scores)
    convex_hull_cohen_kappa_mean_scores.append(mean_cohen_kappa)

########################################################################################################################
# Convex Hull after opening
########################################################################################################################

for i, (img_fname, masks_fnames) in tqdm(enumerate(non_unique_masks_list)):
    masks_fpaths = funcy.walk(lambda x: os.path.join(ISIC_ARCHIVE_TARGETS_PATH, x), masks_fnames)
    masks = funcy.walk(lambda x: load_target_image(x), masks_fpaths)
    resized_masks = funcy.walk(lambda img: transforms.Resize(size=(int(256 * img.size[1] / img.size[0]), 256))(img),
                               masks)
    transformed_masks = funcy.walk(
        lambda img: transform(convex_hull_image, transform(opening, img, selem=selem), mult=255), resized_masks)
    masks_arrays = funcy.walk(lambda x: np.array(x), transformed_masks)
    flattened_masks_arrays = funcy.walk(lambda x: x.flatten(), masks_arrays)

    two_by_two_flattened_masks_arrays = list(combinations(flattened_masks_arrays, 2))
    scores = funcy.walk(lambda t: cohen_kappa_score(*t), two_by_two_flattened_masks_arrays)
    mean_cohen_kappa = np.mean(scores)
    opening_convex_hull_cohen_kappa_mean_scores.append(mean_cohen_kappa)

########################################################################################################################
# Convex Hull after closing
########################################################################################################################

for i, (img_fname, masks_fnames) in tqdm(enumerate(non_unique_masks_list)):
    masks_fpaths = funcy.walk(lambda x: os.path.join(ISIC_ARCHIVE_TARGETS_PATH, x), masks_fnames)
    masks = funcy.walk(lambda x: load_target_image(x), masks_fpaths)
    resized_masks = funcy.walk(lambda img: transforms.Resize(size=(int(256 * img.size[1] / img.size[0]), 256))(img),
                               masks)
    transformed_masks = funcy.walk(
        lambda img: transform(convex_hull_image, transform(closing, img, selem=selem), mult=255), resized_masks)
    masks_arrays = funcy.walk(lambda x: np.array(x), transformed_masks)
    flattened_masks_arrays = funcy.walk(lambda x: x.flatten(), masks_arrays)

    two_by_two_flattened_masks_arrays = list(combinations(flattened_masks_arrays, 2))
    scores = funcy.walk(lambda t: cohen_kappa_score(*t), two_by_two_flattened_masks_arrays)
    mean_cohen_kappa = np.mean(scores)
    closing_convex_hull_cohen_kappa_mean_scores.append(mean_cohen_kappa)

########################################################################################################################
# Bounding Box
########################################################################################################################

for i, (img_fname, masks_fnames) in tqdm(enumerate(non_unique_masks_list)):
    masks_fpaths = funcy.walk(lambda x: os.path.join(ISIC_ARCHIVE_TARGETS_PATH, x), masks_fnames)
    masks = funcy.walk(lambda x: load_target_image(x), masks_fpaths)
    resized_masks = funcy.walk(lambda img: transforms.Resize(size=(int(256 * img.size[1] / img.size[0]), 256))(img),
                               masks)
    transformed_masks = funcy.walk(lambda img: transform(bounding_box, img, mult=255), resized_masks)
    masks_arrays = funcy.walk(lambda x: np.array(x), transformed_masks)
    flattened_masks_arrays = funcy.walk(lambda x: x.flatten(), masks_arrays)

    two_by_two_flattened_masks_arrays = list(combinations(flattened_masks_arrays, 2))
    scores = funcy.walk(lambda t: cohen_kappa_score(*t), two_by_two_flattened_masks_arrays)
    mean_cohen_kappa = np.mean(scores)
    bounding_box_cohen_kappa_mean_scores.append(mean_cohen_kappa)

########################################################################################################################
# Save generated data
########################################################################################################################

data = [
    (fnames[0].split(".")[0],
     original_score,
     opening_score,
     closing_score,
     chull_score,
     opening_chull_score,
     closing_chull_score,
     bounding_box_score)
    for fnames,
        original_score, opening_score, closing_score,
        chull_score, opening_chull_score, closing_chull_score,
        bounding_box_score
    in zip(non_unique_masks_list,
           cohen_kappa_mean_scores,
           opening_cohen_kappa_mean_scores,
           closing_cohen_kappa_mean_scores,
           convex_hull_cohen_kappa_mean_scores,
           opening_convex_hull_cohen_kappa_mean_scores,
           closing_convex_hull_cohen_kappa_mean_scores,
           bounding_box_cohen_kappa_mean_scores)]

df_isic_archive_scores = pd.DataFrame(
    data=sorted(data, key=lambda x: x[1]),
    columns=["lesion_id",
             "original_score",
             "opening_score",
             "closing_score",
             "chull_score",
             "opening_chull_score",
             "closing_chull_score",
             "bounding_box_score"]
)
df_isic_2017_scores = df_isic_archive_scores[df_isic_archive_scores.lesion_id.isin(isic_2017_ids)]
df_isic_2018_scores = df_isic_archive_scores[df_isic_archive_scores.lesion_id.isin(isic_2018_ids)]

df_isic_archive_scores.to_csv(ISIC_ARCHIVE_KAPPA_SCORES_PATH, index=False)
df_isic_2017_scores.to_csv(ISIC_2017_KAPPA_SCORES_PATH, index=False)
df_isic_2018_scores.to_csv(ISIC_2018_KAPPA_SCORES_PATH, index=False)

########################################################################################################################
#
# Visualizing the data
#
########################################################################################################################

########################################################################################################################
# ISIC Archive
########################################################################################################################

fig, ax1 = plt.subplots(figsize=(20, 10))

sns.kdeplot(df_isic_archive_scores["original_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_archive_scores["opening_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_archive_scores["closing_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_archive_scores["chull_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_archive_scores["opening_chull_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_archive_scores["closing_chull_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_archive_scores["bounding_box_score"],
            kernel="gau", bw=0.13, ax=ax1)

ax2 = ax1.twinx()
sns.distplot(df_isic_archive_scores["original_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_archive_scores["opening_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_archive_scores["closing_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_archive_scores["chull_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_archive_scores["opening_chull_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_archive_scores["closing_chull_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_archive_scores["bounding_box_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)

plt.xlim([-0.25, 1.2])
plt.xlabel("Cohen's Kappa Score")
plt.title("ISIC Archive - Cohen's Kappa Score Distribution")
plt.tight_layout()
plt.grid()

plt.savefig(os.path.join(WORKSHOP_FILES_PATH, "isic_archive_cohen_kappa.pdf"))

########################################################################################################################
# ISIC 2017
########################################################################################################################

fig, ax1 = plt.subplots(figsize=(20, 10))

sns.kdeplot(df_isic_2017_scores["original_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_2017_scores["opening_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_2017_scores["closing_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_2017_scores["chull_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_2017_scores["opening_chull_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_2017_scores["closing_chull_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_2017_scores["bounding_box_score"],
            kernel="gau", bw=0.13, ax=ax1)

ax2 = ax1.twinx()
sns.distplot(df_isic_2017_scores["original_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_2017_scores["opening_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_2017_scores["closing_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_2017_scores["chull_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_2017_scores["opening_chull_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_2017_scores["closing_chull_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_2017_scores["bounding_box_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)

plt.xlim([-0.25, 1.2])
plt.xlabel("Cohen's Kappa Score")
plt.title("ISIC 2017 - Cohen's Kappa Score Distribution")
plt.tight_layout()
plt.grid()

plt.savefig(os.path.join(WORKSHOP_FILES_PATH, "isic_2017_cohen_kappa.pdf"))

########################################################################################################################
# ISIC 2018
########################################################################################################################

fig, ax1 = plt.subplots(figsize=(20, 10))

sns.kdeplot(df_isic_2018_scores["original_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_2018_scores["opening_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_2018_scores["closing_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_2018_scores["chull_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_2018_scores["opening_chull_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_2018_scores["closing_chull_score"],
            kernel="gau", bw=0.13, ax=ax1)
sns.kdeplot(df_isic_2018_scores["bounding_box_score"],
            kernel="gau", bw=0.13, ax=ax1)

ax2 = ax1.twinx()
sns.distplot(df_isic_2018_scores["original_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_2018_scores["opening_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_2018_scores["closing_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_2018_scores["chull_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_2018_scores["opening_chull_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_2018_scores["closing_chull_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)
sns.distplot(df_isic_2018_scores["bounding_box_score"],
             hist_kws={"alpha": 0.15}, kde=False, ax=ax2)

plt.xlim([-0.25, 1.2])
plt.xlabel("Cohen's Kappa Score")
plt.title("ISIC 2018 - Cohen's Kappa Score Distribution")
plt.tight_layout()
plt.grid()

plt.savefig(os.path.join(WORKSHOP_FILES_PATH, "isic_2018_cohen_kappa.pdf"))

########################################################################################################################
#
# Statistical significance - Kolmogorov-Smirnov Test
#
########################################################################################################################

dist_combinations = list(combinations(
    ["original_score", "opening_score", "closing_score",
     "chull_score", "opening_chull_score", "closing_chull_score",
     "bounding_box_score"], 2
))
max_len = max([len(f"{first} vs {second}") for first, second in dist_combinations])

isic_archive_ks_data = []
isic_2017_ks_data = []
isic_2018_ks_data = []

print("Kolmogorov-Smirnov Statistical Significance Test")

print("\n")
print("ISIC Archive")
tmp_arr = []
for first, second in dist_combinations:
    sep = "-" * (max_len - len(f"{first} vs {second}") + 4)
    stat, pvalue = ks_2samp(df_isic_archive_scores[first], df_isic_archive_scores[second])
    isic_archive_ks_data.append((first, second, stat, pvalue))
    print(f"{first} vs {second} {sep} pvalue = {pvalue}")

print("\n")
print("ISIC 2017")
tmp_arr = []
for first, second in dist_combinations:
    sep = "-" * (max_len - len(f"{first} vs {second}") + 4)
    stat, pvalue = ks_2samp(df_isic_2017_scores[first], df_isic_archive_scores[second])
    isic_2017_ks_data.append((first, second, stat, pvalue))
    print(f"{first} vs {second} {sep} pvalue = {pvalue}")

print("\n")
print("ISIC 2018")
tmp_arr = []
for first, second in dist_combinations:
    sep = "-" * (max_len - len(f"{first} vs {second}") + 4)
    stat, pvalue = ks_2samp(df_isic_2018_scores[first], df_isic_archive_scores[second])
    isic_2018_ks_data.append((first, second, stat, pvalue))
    print(f"{first} vs {second} {sep} pvalue = {pvalue}")

df_isic_archive_ks = pd.DataFrame(data=isic_archive_ks_data, columns=["transform_1", "transform_2", "stat", "pvalue"])
df_isic_2017_ks = pd.DataFrame(data=isic_2017_ks_data, columns=["transform_1", "transform_2", "stat", "pvalue"])
df_isic_2018_ks = pd.DataFrame(data=isic_2018_ks_data, columns=["transform_1", "transform_2", "stat", "pvalue"])

df_isic_archive_ks.to_csv(os.path.join(WORKSHOP_FILES_PATH, "isic_archive_ks.csv"), index=False)
df_isic_2017_ks.to_csv(os.path.join(WORKSHOP_FILES_PATH, "isic_2017_ks.csv"), index=False)
df_isic_2018_ks.to_csv(os.path.join(WORKSHOP_FILES_PATH, "isic_2018_ks.csv"), index=False)
