import torch
import torch.nn as nn
import random
import torchvision.transforms.functional as F


class ColorJitterCustom:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, item):
        # Split the image into the first 6 channels and the 7th channel
        img = item["data"]
        labels = item["labels"]
        img_max = img.max()
        first_3_channels = img[:3, :, :] / img_max
        other_3_channels = img[3:6, :, :] / img_max
        seventh_channel = labels

        brightness_factor = 1 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1 + random.uniform(-self.contrast, self.contrast)
        saturation_factor = 1 + random.uniform(-self.saturation, self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        # Apply color jitter to the first 3 channels
        jittered_channels0 = F.adjust_brightness(
            first_3_channels, brightness_factor
        )  # there is no adjust_brightness in the official PyTorch
        jittered_channels0 = F.adjust_contrast(jittered_channels0, contrast_factor)
        jittered_channels0 = F.adjust_saturation(jittered_channels0, saturation_factor)
        jittered_channels0 = F.adjust_hue(jittered_channels0, hue_factor)
        # Apply color jitter to the other 3 channels
        jittered_channels1 = F.adjust_brightness(other_3_channels, brightness_factor)
        jittered_channels1 = F.adjust_contrast(jittered_channels1, contrast_factor)
        jittered_channels1 = F.adjust_saturation(jittered_channels1, saturation_factor)
        jittered_channels1 = F.adjust_hue(jittered_channels1, hue_factor)

        # Concatenate the jittered channels with the 7th channel
        item["data"] = torch.cat(
            (jittered_channels0 * img_max, jittered_channels1 * img_max), dim=0
        )
        item["labels"] = seventh_channel
        return item


# Custom transformations
def random_rotation(item):
    img = item["data"]
    labels = item["labels"]
    angle = torch.randint(0, 4, (1,)).item()
    img = torch.tensor(img)
    labels = torch.tensor(labels)
    img = F.rotate(img, angle * 90)
    labels = F.rotate(labels, angle * 90)
    return {"data": img, "labels": labels}


def random_horizontal_flip(item):
    img = item["data"]
    labels = item["labels"]
    img = torch.tensor(img)
    labels = torch.tensor(labels)
    if torch.rand(1).item() > 0.5:
        return {"data": F.hflip(img), "labels": F.hflip(labels)}
    return item


def random_vertical_flip(item):
    img = item["data"]
    labels = item["labels"]
    img = torch.tensor(img)
    labels = torch.tensor(labels)
    if torch.rand(1).item() > 0.5:
        return {"data": F.vflip(img), "labels": F.vflip(labels)}
    return item
