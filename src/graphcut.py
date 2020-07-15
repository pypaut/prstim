import matplotlib.image as mpimg
import numpy as np
from collections import defaultdict


def rgb2yiq(img):
    yiq = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = img[i][j][0]
            g = img[i][j][1]
            b = img[i][j][2]
            yiq[i][j][0] = 0.2990 * r + 0.5870 * g + 0.1140 * b
            yiq[i][j][1] = 0.5959 * r - 0.2746 * g - 0.3213 * b
            yiq[i][j][2] = 0.2115 * r - 0.5227 * g + 0.3112 * b
    return yiq


def find_marked_locations(rgb, rgb_s):
    locations = []
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            if rgb[i][j].all() != rgb_s[i][j].all():
                locations.append((i, j))
    return locations


def compute_pdfs(imfile, imfile_scrib):
    """
    # Compute foreground and background pdfs
    # input image and the image with user scribbles
    """
    rgb = mpimg.imread(imfile)[:, :, :3]
    print(rgb.shape)
    yuv = rgb2yiq(rgb)
    rgb_s = mpimg.imread(imfile_scrib)[:, :, :3]
    yuv_s = rgb2yiq(rgb_s)
    # find the scribble pixels
    scribbles = find_marked_locations(rgb, rgb_s)
    imageo = np.zeros(yuv.shape)
    # separately store background and foreground scribble pixels in the dictionary comps
    comps = defaultdict(lambda: np.array([]).reshape(0, 3))
    for (i, j) in scribbles:
        imageo[i, j, :] = rgb_s[i, j, :]
        # scribble color as key of comps
        comps[tuple(imageo[i, j, :])] = np.vstack(
            [comps[tuple(imageo[i, j, :])], yuv[i, j, :]]
        )
        mu, Sigma = {}, {}
    # compute MLE parameters for Gaussians
    for c in comps:
        mu[c] = np.mean(comps[c], axis=0)
        Sigma[c] = np.cov(comps[c].T)
    return (mu, Sigma)


res = compute_pdfs(
    "notebook/data/Venice_2019.png", "notebook/data/Venice_2019_scrib.png"
)
print(res)
