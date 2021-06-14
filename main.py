# importing required libraries
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

# a flag, weather to write results to disk or not
my_flag = 1


def writetodisk(path, img, flag):
    if flag == 1:
        cv.imwrite(path, img)


def merge(r_channel, g_channel, b_channel):
    rows = r_channel.shape[0]
    columns = r_channel.shape[1]
    out_img = np.zeros((rows, columns, 3))
    out_img[:, :, 0] = r_channel
    out_img[:, :, 1] = g_channel
    out_img[:, :, 2] = b_channel
    return out_img


# reading image
img_1 = cv.imread("cameraman.png")
img_shape = img_1.shape

# separating Red, Green and Blue Channels
R_channel = img_1[:, :, 0]
G_channel = img_1[:, :, 1]
B_channel = img_1[:, :, 2]

# saving results
writetodisk("./Problem_01_results/r_channel.png", R_channel, my_flag)
writetodisk("./Problem_01_results/r_channel.png", R_channel, my_flag)
writetodisk("./Problem_01_results/g_channel.png", G_channel, my_flag)
writetodisk("./Problem_01_results/b_channel.png", B_channel, my_flag)

# given filters
h_1 = 1 / 9 * np.ones((3, 3))
h_2 = 1 / 256 * np.array([[1, 4, 6, 4, 1],
                          [4, 16, 24, 16, 4],
                          [6, 24, 36, 24, 6],
                          [4, 16, 24, 16, 4],
                          [1, 4, 6, 4, 1]
                          ])
# applying filter
R_channel_flt_1 = cv.filter2D(R_channel, -1, h_1)
G_channel_flt_1 = cv.filter2D(G_channel, -1, h_1)
B_channel_flt_1 = cv.filter2D(B_channel, -1, h_1)

R_channel_flt_2 = cv.filter2D(R_channel, -1, h_2)
G_channel_flt_2 = cv.filter2D(G_channel, -1, h_2)
B_channel_flt_2 = cv.filter2D(B_channel, -1, h_2)

# merging back
flt_img_1 = merge(R_channel_flt_1, G_channel_flt_1, B_channel_flt_1)
flt_img_2 = merge(R_channel_flt_2, G_channel_flt_2, B_channel_flt_2)

# saving results
writetodisk("./Problem_01_results/P_01_flt_img_01.png", flt_img_1, my_flag)
writetodisk("./Problem_01_results/P_01_flt_img_02.png", flt_img_2, my_flag)


# #######################################################################################
# Problem: 02
def getMSE(img_1, img_2):
    if img_1.shape == img_2.shape:
        dia = img_1.shape
        mse = (1 / (dia[0] * dia[1] * dia[2])) * np.sum(np.square(img_1 - img_2))
        return mse
    else:
        print("Images should have same dimension")


def getPSNR(max, mse):
    return 10 * math.log10((np.square(max)) / mse)


mse_img = getMSE(img_1, flt_img_1)
psnr_img = getPSNR(255, mse_img)

######################################################################################
