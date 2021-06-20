# importing required libraries
import cv2 as cv
import numpy as np
import math
import time
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


p_01_start_time = time.time()
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

p_01_end_time = time.time()

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


# starting time of problem 02
p_02_start_time = time.time()

mse_img = getMSE(img_1, flt_img_1)
psnr_img = getPSNR(255, mse_img)

# end time of the problem 02
p_02_end_time = time.time()
######################################################################################
# Problem 03
# converting image to hsi
hsi_img = cv.cvtColor(img_1, cv.COLOR_RGB2Lab)

# separating H, S and I part
h_channel = hsi_img[:, :, 0]
s_channel = hsi_img[:, :, 1]
i_channel = hsi_img[:, :, 2]

# filtering I only
i_flt_img_1 = cv.filter2D(i_channel, -1, h_1)
i_flt_img_2 = cv.filter2D(i_channel, -1, h_2)

# merging back
hsi_flt_img_1 = merge(h_channel, s_channel, i_flt_img_1)
hsi_flt_img_2 = merge(h_channel, s_channel, i_flt_img_2)

# converting from float64 type to float32
# as float64 is not supported by cv.cvtColor() function
hsi_flt_img_1 = np.float32(hsi_flt_img_1)
hsi_flt_img_2 = np.float32(hsi_flt_img_2)

# converting back to rgb
hsi2rgb_img_1 = cv.cvtColor(hsi_flt_img_1, cv.COLOR_HSV2RGB)
hsi2rgb_img_2 = cv.cvtColor(hsi_flt_img_2, cv.COLOR_HSV2RGB)

# finding mse value
mse_img_1 = getMSE(img_1, hsi2rgb_img_1)
mse_img_2 = getMSE(img_1, hsi2rgb_img_2)

# finding psnr value
psnr_img_1 = getPSNR(255, mse_img_1)
psnr_img_2 = getPSNR(255, mse_img_2)

######################################################################################
# problem 04

# starting time of problem 04
p_04_start_time = time.time()

# converting RGB space to YCbCr
ycbcr_img = cv.cvtColor(img_1, cv.COLOR_RGB2YCrCb)

# separating channels
y_channel = ycbcr_img[:, :, 0]
cb_channel = ycbcr_img[:, :, 1]
cr_channel = ycbcr_img[:, :, 2]

# applying filers to y channel
flt_01_y_channel = cv.filter2D(y_channel, -1, h_1)
flt_02_y_channel = cv.filter2D(y_channel, -1, h_2)

# merging back
flt_ycbcr_img_01 = merge(flt_01_y_channel, cb_channel, cr_channel)
flt_ycbcr_img_02 = merge(flt_02_y_channel, cb_channel, cr_channel)

# converting float 64 to float 32
flt_ycbcr_img_01 = np.float32(flt_ycbcr_img_01)
flt_ycbcr_img_02 = np.float32(flt_ycbcr_img_02)

# taking back to RGB space
flt_rgb_img_01 = cv.cvtColor(flt_ycbcr_img_01, cv.COLOR_YCrCb2RGB)
flt_rgb_img_02 = cv.cvtColor(flt_ycbcr_img_02, cv.COLOR_YCrCb2RGB)

# ending time of problem 04
p_04_end_time = time.time()
