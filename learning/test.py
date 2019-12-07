import argparse
import funcy
import json
import numpy as np
import os
import random
import torch

from collections import OrderedDict
from skimage.morphology import square
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SkinLesionSegmentationDataset
from loss import SoftJaccardBCEWithLogitsLoss, evaluate_jaccard, evaluate_dice
from models.autodeeplab.auto_deeplab import AutoDeeplab
from models.deeplab.deeplab import DeepLab
from models.linknet import LinkNet
from models.refinenet.refinenet_4cascade import RefineNet4Cascade
from models.unet import UNet11
from transforms.target import Opening, ConvexHull, BoundingBox

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

available_conditioning = {
    "original": lambda x: x,
    "opening": Opening(square, 5),
    "convex_hull": funcy.rcompose(Opening(square, 5), ConvexHull()),
    "bounding_box": funcy.rcompose(Opening(square, 5), BoundingBox()),
}


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def run_test(model, dataloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress_bar = tqdm(dataloader, desc="Testing")

    model.eval()

    losses = []
    jaccards = []
    jaccards_threshold = []
    dices = []
    for i, (inputs, targets, fname, (_, _)) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            jaccard = evaluate_jaccard(outputs, targets)
            jaccard_threshold = jaccard.item() if jaccard.item() > 0.65 else 0.0
            dice = evaluate_dice(jaccard.item())

            losses.append(loss.item())
            jaccards.append(jaccard.item())
            jaccards_threshold.append(jaccard_threshold)
            dices.append(dice)
            progress_bar.set_postfix(OrderedDict({"loss": np.mean(losses),
                                                  "jaccard": np.mean(jaccards),
                                                  "jaccard_threshold": np.mean(jaccards_threshold),
                                                  "dice": np.mean(dices)}))

    mean_loss = np.mean(losses)
    mean_jacc = np.mean(jaccards)
    mean_jacc_threshold = np.mean(jaccards_threshold)
    mean_dice = np.mean(dices)

    info = {"loss": mean_loss,
            "jaccard": mean_jacc,
            "jaccard_threshold": mean_jacc_threshold,
            "dice": mean_dice}

    return info


def main(data_name, fpath, model_path, model, save_to=None, conditioning="original"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conditioning_fn = available_conditioning[conditioning]
    dataset = SkinLesionSegmentationDataset(fpath, target_preprocess=conditioning_fn)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False, worker_init_fn=set_seeds)
    loss_fn = SoftJaccardBCEWithLogitsLoss(jaccard_weight=8)

    if model == "deeplab":
        model = DeepLab(num_classes=1).to(device)
    elif model == "autodeeplab":
        model = AutoDeeplab(num_classes=1).to(device)
    elif model == "unet":
        model = UNet11(pretrained=True).to(device)
    elif model == "linknet":
        model = LinkNet(n_classes=1).to(device)
    elif model == "refinenet":
        model = RefineNet4Cascade(input_shape=(3, 256)).to(device)  # 3 channels, 256x256 input
    else:
        raise Exception("Invalid model '{}'".format(model))

    model.load_state_dict(torch.load(model_path))

    info = run_test(model, dataloader, loss_fn)

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    with open(os.path.join(save_to, "test_{}_{}.json".format(data_name, conditioning)), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model")
    parser.add_argument("--model-path", dest="model_path")
    parser.add_argument("--save-to", dest="save_to")
    parser.add_argument("--conditioning", dest="conditioning", default="original")
    args = parser.parse_args()

    assert args.conditioning in available_conditioning, "Unavailable conditioning '{}'".format(args.conditioning)

    test_dermofit_path = os.path.join(BASE_PATH, "data", "test_dermofit.txt")
    test_isic_titans_path = os.path.join(BASE_PATH, "data", "test_isic_titans_2000.txt")
    test_ph2_path = os.path.join(BASE_PATH, "data", "test_ph2.txt")

    main("dermofit", test_dermofit_path, args.model_path, args.model, args.save_to, args.conditioning)
    main("isic_titans", test_isic_titans_path, args.model_path, args.model, args.save_to, args.conditioning)
    main("ph2", test_ph2_path, args.model_path, args.model, args.save_to, args.conditioning)
