import cv2
import matplotlib.pyplot as plt
import numpy as np

#############
### ANNEX ###
#############


def preprocess(img):
    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img.astype(np.int8)  # Allow negatives, for diff

    # Histogram equalization (color)
    # yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    # equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    # return equalized


def difference(im2, im1):
    diff = np.absolute(im2 - im1)
    # diff = cv2.equalizeHist(diff)

    # For RGB : Euclidian distance between pixels
    # diff = im2 - im1
    # diff = np.apply_along_axis(np.linalg.norm, axis=2, arr=diff)

    return diff


def threshold(diff):
    thresh = 100
    return diff > 100


############
### MAIN ###
############


def main():
    """
    Goal : count boats.
    Provided data : twice the same picture, with a one-year gap.
    Method : we know that boats are part of the differences between the two
    pictures (not likely to find a boat at the same location twice).
    We will thus work on these differences, and then distinguish boats from
    the rest of the shapes.
    """
    # Load images
    im1, im2 = (
        cv2.imread("data/Venice_2019.png"),
        cv2.imread("data/Venice_2020.png"),
    )

    # Preprocess
    im1, im2 = (
        preprocess(im1),
        preprocess(im2),
    )

    # Compute difference
    diff = difference(im2, im1)
    plt.imsave("output/eqdiff.png", diff, cmap="gray")

    # Threshold difference
    thresh = threshold(diff)


if __name__ == "__main__":
    main()
