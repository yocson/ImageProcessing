from skimage import io
import numpy as np

# read images
image1 = io.imread("NoisyImage1.jpg")
image2 = io.imread("NoisyImage2.jpg")

# two kernel
k1 = np.divide(np.ones((3,3)), 9)
k2 = np.divide(np.ones((5,5)), 25)
