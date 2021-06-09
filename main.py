# importing required libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# reading image
img_1 = cv.imread("cameraman.png")
img_shape = img_1.shape

# separating Red, Green and Blue Channels
R_channel = img_1[:, :, 0]
G_channel = img_1[:, :, 1]
B_channel = img_1[:, :, 2]

# saving results
cv.imwrite("./Problem_01_results/r_channel.png", R_channel)
cv.imwrite("./Problem_01_results/g_channel.png", G_channel)
cv.imwrite("./Problem_01_results/b_channel.png", B_channel)

# given filters
h_1 = 1 / 9 * np.ones((3, 3))
h_2 = 1 / 256 * np.array([[1, 4, 6, 4, 1],
                          [4, 16, 24, 16, 4],
                          [6, 24, 36, 24, 6],
                          [4, 16, 24, 16, 4],
                          [1, 4, 6, 4, 1]
                          ])

