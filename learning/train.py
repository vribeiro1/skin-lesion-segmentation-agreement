import funcy
import numpy as np
import os
import random
import torch

from sacred import Experiment
from sacred.observers import FileStorageObserver
from skimage.morphology import square
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

from dataset import SkinLesionSegmentationDataset, MultimaskSkinLesionSegmentationDataset
from loss import SoftJaccardBCEWithLogitsLoss, evaluate_jaccard, evaluate_dice
from models.autodeeplab.auto_deeplab import AutoDeeplab
from models.deeplab.deeplab import DeepLab
from models.linknet import LinkNet
from models.refinenet.refinenet_4cascade import RefineNet4Cascade
from models.unet import UNet11
from transforms.input import GaussianNoise, EnhanceContrast, EnhanceColor
from transforms.target import Opening, ConvexHull, BoundingBox

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

ex = Experiment()
fs_observer = FileStorageObserver.create(os.path.join(BASE_PATH, "results"))
ex.observers.append(fs_observer)

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


def run_epoch(phase, epoch, model, dataloader, optimizer, criterion, scheduler=None, writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    progress_bar = tqdm(dataloader, desc="Epoch {} - {}".format(epoch, phase))
    training = phase == "train"

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    jaccards = []
    jaccards_threshold = []
    dices = []
    for i, (inputs, targets, fname, (_, _)) in enumerate(progress_bar):
        inputs = inputs.to(device)
        if isinstance(targets, list):
            targets = funcy.walk(lambda target: target.to(device), targets)
        else:
            targets = targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(inputs)

            if isinstance(targets, list):
                loss = min(funcy.walk(lambda target: criterion(outputs, target), targets))
                jaccard = max(funcy.walk(lambda target: evaluate_jaccard(outputs, target), targets))
            else:
                loss = criterion(outputs, targets)
                jaccard = evaluate_jaccard(outputs, targets)
            jaccard_threshold = jaccard.item() if jaccard.item() > 0.65 else 0.0
            dice = evaluate_dice(jaccard.item())

            if training:
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.batch_step()

            losses.append(loss.item())
            jaccards.append(jaccard.item())
            jaccards_threshold.append(jaccard_threshold)
            dices.append(dice)
            progress_bar.set_postfix(loss=np.mean(losses),
                                     jaccard=np.mean(jaccards),
                                     jaccard_threshold=np.mean(jaccards_threshold),
                                     dice=np.mean(dices))

    mean_loss = np.mean(losses)
    loss_tag = "{}.loss".format(phase)
    writer.add_scalar(loss_tag, mean_loss, epoch)

    mean_jacc = np.mean(jaccards)
    jacc_tag = "{}.jaccard".format(phase)
    writer.add_scalar(jacc_tag, mean_jacc, epoch)

    mean_jacc_thr = np.mean(jaccards_threshold)
    jacc_thr_tag = "{}.jaccard_threshold".format(phase)
    writer.add_scalar(jacc_thr_tag, mean_jacc_thr, epoch)

    mean_dice = np.mean(dices)
    dice_tag = "{}.dice".format(phase)
    writer.add_scalar(dice_tag, mean_dice, epoch)

    info = {"loss": mean_loss,
            "jaccard": mean_jacc,
            "jaccard_threshold": mean_jacc_thr,
            "dice": mean_dice}

    return info


@ex.automain
def main(_run, model, batch_size, n_epochs, lr, multimask, patience,
         train_fpath, val_fpath, train_conditioning, val_conditioning,
         model_fpath=None):
    run_validation = val_fpath is not None

    err_msg = "{} conditioning '{}' is not available. Available functions are: '{}'"
    assert train_conditioning in available_conditioning, err_msg.format("Train", train_conditioning,
                                                                        list(available_conditioning.keys()))
    if run_validation:
        assert val_conditioning in available_conditioning, err_msg.format("Validation", val_conditioning,
                                                                          list(available_conditioning.keys()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(os.path.join(BASE_PATH, "runs", "experiment-{}".format(_run._id)))
    best_model_path = os.path.join(fs_observer.dir, "best_model.pth")
    last_model_path = os.path.join(fs_observer.dir, "last_model.pth")

    outputs_path = os.path.join(fs_observer.dir, "outputs")
    if not os.path.exists(outputs_path):
        os.mkdir(outputs_path)

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

    if model_fpath is not None:
        state_dict = torch.load(model_fpath)
        model.load_state_dict(state_dict)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = SoftJaccardBCEWithLogitsLoss(jaccard_weight=8)

    augmentations = [
        GaussianNoise(0, 2),
        EnhanceContrast(0.5, 0.1),
        EnhanceColor(0.5, 0.1)
    ]

    dataloaders = {}
    train_preprocess_fn = available_conditioning[train_conditioning]
    val_preprocess_fn = available_conditioning[val_conditioning]

    train_dataset_args = dict(
        fpath=train_fpath,
        augmentations=augmentations,
        target_preprocess=train_preprocess_fn
    )
    validation_dataset_args = dict(
        fpath=val_fpath,
        target_preprocess=val_preprocess_fn
    )

    if multimask:
        DatasetClass = MultimaskSkinLesionSegmentationDataset
        train_dataset_args["select"] = "random"
        validation_dataset_args["select"] = "all"
    else:
        DatasetClass = SkinLesionSegmentationDataset

    train_dataset = DatasetClass(**train_dataset_args)
    dataloaders["train"] = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      num_workers=8,
                                      shuffle=True,
                                      worker_init_fn=set_seeds)

    if run_validation:
        val_dataset = DatasetClass(**validation_dataset_args)
        dataloaders["validation"] = DataLoader(val_dataset,
                                               batch_size=batch_size if not multimask else 1,
                                               num_workers=8,
                                               shuffle=False,
                                               worker_init_fn=set_seeds)

    info = {}
    epochs = range(1, n_epochs + 1)
    best_jacc = 0
    epochs_since_best = 0

    for epoch in epochs:
        info["train"] = run_epoch("train", epoch, model, dataloaders["train"], optimizer, loss_fn, writer=writer)

        if run_validation:
            info["validation"] = run_epoch("validation", epoch, model, dataloaders["validation"], optimizer, loss_fn, writer=writer)
            if info["validation"]["jaccard"] > best_jacc:
                best_jacc = info["validation"]["jaccard"]
                torch.save(model.state_dict(), best_model_path)
                epochs_since_best = 0
            else:
                epochs_since_best += 1

        torch.save(model.state_dict(), last_model_path)

        if epochs_since_best > patience:
            break
