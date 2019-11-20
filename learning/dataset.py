import os
import funcy
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Normalize
from typing import Callable, List, Tuple
from PIL import Image


class SkinLesionSegmentationDataset(Dataset):
    def __init__(self, fpath: str, augmentations: List = None, input_preprocess: Callable = None,
                 target_preprocess: Callable = None, with_targets: bool = True, shape: Tuple = (256, 256)):
        if not os.path.isfile(fpath):
            raise FileNotFoundError("Could not find dataset file: '{}'".format(fpath))
        self.with_targets = with_targets
        self.size = shape

        if augmentations:
            augmentations = [lambda x: x] + augmentations
        else:
            augmentations = [lambda x: x]

        self.resize = Resize(size=self.size)
        # self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.normalize = Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
        self.to_tensor = ToTensor()
        self.input_preprocess = input_preprocess
        self.target_preprocess = target_preprocess

        with open(fpath, "r") as f:
            lines = filter(lambda l: bool(l), f.read().split("\n"))
            if self.with_targets:
                data = [(input.strip(), target.strip())
                        for input, target in funcy.walk(lambda l: l.split(" "), lines)]
            else:
                data = [(input.strip(), None) for input in lines]

        self.data = [(d, augmentation) for augmentation in augmentations for d in data]

    @staticmethod
    def _load_input_image(fpath: str):
        img = Image.open(fpath).convert("RGB")
        return img

    @staticmethod
    def _load_target_image(fpath: str):
        img = Image.open(fpath).convert("L")
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        (input_fpath, target_fpath), augmentation = self.data[item]

        input_img = self._load_input_image(input_fpath)
        width, height = input_img.size
        input_img = self.resize(input_img)

        if self.input_preprocess is not None:
            input_img = self.input_preprocess(input_img)

        input_img = augmentation(input_img)
        input_img = self.to_tensor(input_img)
        input_img = self.normalize(input_img)

        target_img = ""
        if target_fpath is not None:
            target_img = self._load_target_image(target_fpath)
            target_img = self.resize(target_img)

            if self.target_preprocess is not None:
                target_img = self.target_preprocess(target_img)

            target_img = self.to_tensor(target_img)

        fname = os.path.basename(input_fpath).split(".")[0]

        return input_img, target_img, fname, (width, height)


class MultimaskSkinLesionSegmentationDataset(Dataset):
    def __init__(self, fpath: str, augmentations: List = None, input_preprocess: Callable = None,
                 target_preprocess: Callable = None, with_targets: bool = True, select="random",
                 shape: Tuple = (256, 256)):
        if not os.path.isfile(fpath):
            raise FileNotFoundError("Could not find dataset file: '{}'".format(fpath))

        self.with_targets = with_targets
        self.size = shape

        if augmentations:
            augmentations = [lambda x: x] + augmentations
        else:
            augmentations = [lambda x: x]

        self.resize = Resize(size=self.size)
        # self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.normalize = Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
        self.to_tensor = ToTensor()
        self.input_preprocess = input_preprocess
        self.target_preprocess = target_preprocess

        with open(fpath, "r") as f:
            lines = filter(lambda l: bool(l), f.read().split("\n"))
            if self.with_targets:
                data = []
                for line in lines:
                    fpaths = line.split(" ")
                    input_ = fpaths[0].strip()
                    targets = funcy.walk(lambda f: f.strip(), fpaths[1:])

                    data.append(
                        (input_, targets)
                    )
            else:
                data = [(input.strip(), None) for input in lines]

        if select == "all":
            self.selection_method = self._select_all
        else:
            self.selection_method = self._random_selection

        self.data = [(d, augmentation) for augmentation in augmentations for d in data]

    @staticmethod
    def _load_input_image(fpath: str):
        img = Image.open(fpath).convert("RGB")
        return img

    @staticmethod
    def _load_target_image(fpath: str):
        img = Image.open(fpath).convert("L")
        return img

    def _random_selection(self, targets_list: List[str]):
        target_fpath = np.random.choice(targets_list)

        target_img = self._load_target_image(target_fpath)
        target_img = self.resize(target_img)

        if self.target_preprocess is not None:
            target_img = self.target_preprocess(target_img)

        return [target_img]

    def _select_all(self, targets_list: List[str]):
        target_imgs = []

        for target_fpath in targets_list:
            target_img = self._load_target_image(target_fpath)
            target_img = self.resize(target_img)

            if self.target_preprocess is not None:
                target_img = self.target_preprocess(target_img)

            target_imgs.append(target_img)

        return target_imgs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        (input_fpath, targets_fpaths), augmentation = self.data[item]

        input_img = self._load_input_image(input_fpath)
        width, height = input_img.size
        input_img = self.resize(input_img)

        if self.input_preprocess is not None:
            input_img = self.input_preprocess(input_img)

        input_img = augmentation(input_img)
        input_img = self.to_tensor(input_img)
        input_img = self.normalize(input_img)

        target_imgs = None
        if self.with_targets:
            target_imgs = self.selection_method(targets_fpaths)
            target_imgs = funcy.walk(self.to_tensor, target_imgs)

        fname = os.path.basename(input_fpath).split(".")[0]

        if self.selection_method == self._random_selection:
            target_imgs = target_imgs[0]

        return input_img, target_imgs, fname, (width, height)
