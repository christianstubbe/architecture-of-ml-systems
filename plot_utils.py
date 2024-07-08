import os
import numpy as np
import matplotlib.pyplot as plt

def stretch_hist(band, p_low=0.5, p_high=99.5, indiv_bands=False):
    """
    Apply histogram stretching"""

    p_down, p_up = np.percentile(band, (p_low, p_high))
    # print("p_down, p_up",p_down, p_up)
    return np.clip((band - p_down) * 1 / (p_up - p_down), 0, 1).astype(np.float32)

        
def histogram_scaler_bands(bands, p_low=0.5, p_high=99.5):
    """
    Apply histogram scaling
    THIS IS THE BEST WAY TO SCALE THE BANDS!
    takes in a np.ndarray of shape (n_bands, rows, cols) and returns the indiv. scaled bands
    """
    p_down, p_up = np.percentile(bands, (p_low, p_high), axis=(1, 2))
    print("p_down, p_up",p_down, p_up)
    return np.clip((bands - p_down[:, None, None]) / (p_up - p_down)[:, None, None], 0, 1).astype(np.float32)
    


def histogram_scale(band, p_low=0.5, p_high=99.5, indiv_bands=False):
    """
    Apply histogram scaling"""
    # if indiv_bands:
    #     p_down, p_up = np.percentile(band, (p_low, p_high))
    # else:
    #     p_down, p_up = np.percentile(band.flatten(), (p_low, p_high))
    # return np.clip((band - p_down) / (p_up - p_down), 0, 1).astype(np.float32)
    p_down, p_up = np.percentile(band, (p_low, p_high))
    print("p_down, p_up",p_down, p_up)
    return np.clip((band - p_down) / (p_up - p_down), 0, 1).astype(np.float32)

def plot_band(band, title="Single Band", cmap="Blues", show_axis=False):
    """
    Plot a band
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap=cmap)
    plt.title(title)
    if not show_axis:
        plt.axis("off")
    plt.show()

def plot_bands(data, bands=[1], title="Multi Plots",  show_axis=False):
    """
    Plot a band
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(np.stack([data[b] for b in bands], axis=-1))
    plt.title(title)
    if not show_axis:
        plt.axis("off")
    plt.show()

def plot_band_with_mask(band, mask, title="Band with Mask", cmap_band="grey", cmap_mask="Blues", show_axis=False):
    """
    Plot a single band with mask overlay
    """

    plt.figure(figsize=(10, 10))
    plt.imshow(band, cmap=cmap_band)
    plt.imshow(mask, cmap=cmap_mask, alpha=0.5)
    plt.title(title)
    if not show_axis:
        plt.axis("off")
    plt.show()

def describe_tif(src):
    print("Profile:\n\t",src.profile)
    print("SHAPE:\t\t",src.read(1).shape)
    print("dtype\t\t", src.read(1).dtype)
    print("max\t\t",src.read(1).max())
    print("min\t\t",src.read(1).min())
    print("mean\t\t", src.read(1).mean())
    print("std\t\t",src.read(1).std())
    print("sum\t\t", src.read(1).sum())


def plot_prediction_with_thresholds(output: np.ndarray,
                                    thresholds: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9],
                                    save_path=None):
    fig, axs = plt.subplots(5, 2, figsize=(10, 20))
    axs = axs.flatten()

    for i, threshold in enumerate(thresholds):
        axs[i].imshow(output > threshold)
        axs[i].set_title(f'Threshold: {threshold}')
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(os.path.join(save_path, "thresholds.png"))


def plot_random_patch(output: np.ndarray, patch_len: int=8, save_path=None):
    x = np.random.randint(0, output.shape[0]-patch_len)
    y = np.random.randint(0, output.shape[1]-patch_len)
    patch = output[x:x+patch_len, y:y+patch_len]
    plt.imshow(patch)
    plt.show()
    print(patch)
    if save_path:
        plt.savefig(os.path.join(save_path, "rand_patch.png"))

def plot_output(output: np.ndarray, save_path=None):
    # plot the output
    plt.figure(figsize=(10,10))
    plt.imshow(output)
    plt.show()
    if save_path:
        plt.savefig(os.path.join(save_path, "output.png"))