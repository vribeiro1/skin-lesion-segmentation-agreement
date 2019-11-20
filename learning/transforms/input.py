import numpy as np

from PIL import Image, ImageEnhance


def gaussian_noise(img, mean, std):
    img_arr = np.array(img)

    h, w, c = img_arr.shape
    noise = np.random.normal(mean, std, (h, w, c))

    new_img_arr = img_arr + noise
    new_img = Image.fromarray(new_img_arr.astype(np.uint8))

    return new_img


def enhance_brightness(img, brightness_factor):
    enhancer = ImageEnhance.Brightness(img)
    enhanced_img = enhancer.enhance(brightness_factor)

    return enhanced_img


def enhance_contrast(img, contrast_factor):
    enhancer = ImageEnhance.Contrast(img)
    enhanced_img = enhancer.enhance(contrast_factor)

    return enhanced_img


def enhance_color(img, color_factor):
    enhancer = ImageEnhance.Color(img)
    enhanced_img = enhancer.enhance(color_factor)

    return enhanced_img


def enhance_sharpness(img, sharpness_factor):
    enhancer = ImageEnhance.Sharpness(img)
    enhanced_img = enhancer.enhance(sharpness_factor)

    return enhanced_img


def color_gradient(img, gradient_factor, mode, color):
    mode_multipliers = {
        "upper_left": (1, 1),
        "upper_right": (-1, 1),
        "bottom_left": (1, -1),
        "bottom_right": (-1, -1),
        "upper": (0, 1),
        "bottom": (0, -1),
        "left": (1, 0),
        "right": (-1, 0)
    }

    i_mult, j_mult = mode_multipliers[mode]

    img_arr = np.array(img)
    h, w, c = img_arr.shape

    gradient_img_arr = np.ones((h, w))

    if i_mult == 0:
        for j in range(h):
            value = gradient_factor * 255 * j / h
            gradient_img_arr[j_mult * j, :] = value
    elif j_mult == 0:
        for i in range(w):
            value = gradient_factor * 255 * i / w
            gradient_img_arr[:, i_mult * i] = value
    else:
        for i in range(w):
            for j in range(h):
                value = gradient_factor * 255 * (i + j) / (w + h)
                gradient_img_arr[j_mult * j][i_mult * i] = value

    gradient = Image.fromarray(gradient_img_arr.astype(np.uint8))
    gradient = gradient.convert("L")

    black_img = Image.new("RGBA", (w, h), color=color)
    black_img.putalpha(gradient)

    new_img = Image.alpha_composite(img.convert("RGBA"), black_img)

    return new_img


class GaussianNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return gaussian_noise(img, self.mean, self.std)


class EnhanceBrightness:
    def __init__(self, factor_mean, factor_std):
        self.factor_mean = factor_mean
        self.factor_std = factor_std

    def __call__(self, img):
        factor = np.random.normal(self.factor_mean, self.factor_std)
        return enhance_brightness(img, factor)


class EnhanceContrast:
    def __init__(self, factor_mean, factor_std):
        self.factor_mean = factor_mean
        self.factor_std = factor_std

    def __call__(self, img):
        factor = np.random.normal(self.factor_mean, self.factor_std)
        return enhance_contrast(img, factor)


class EnhanceColor:
    def __init__(self, factor_mean, factor_std):
        self.factor_mean = factor_mean
        self.factor_std = factor_std

    def __call__(self, img):
        factor = np.random.normal(self.factor_mean, self.factor_std)
        return enhance_color(img, factor)


class EnhanceSharpness:
    def __init__(self, factor_mean, factor_std):
        self.factor_mean = factor_mean
        self.factor_std = factor_std

    def __call__(self, img):
        factor = np.random.normal(self.factor_mean, self.factor_std)
        return enhance_sharpness(img, factor)


class ColorGradient:
    MODES = [
        "upper_left",
        "upper_right",
        "bottom_left",
        "bottom_right",
        "upper",
        "bottom",
        "left",
        "right"
    ]

    def __init__(self, factor=None, mode=None, color=0x000000):
        if mode is not None and mode not in self.MODES:
            raise Exception("Unavailable mode '{}'. Available modes are {}".format(mode, self.MODES))

        self.factor = factor
        self.mode = mode
        self.color = color

    def __call__(self, img):
        factor = self.factor if self.factor is not None else np.random.uniform()
        mode = self.mode if self.mode is not None else np.random.choice(self.MODES)

        return color_gradient(img, factor, mode, self.color)
