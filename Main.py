from __future__ import print_function
import cv2
import numpy as np
import math
import argparse
import os
import time
import random
import maxflow
from PIL import Image
import sys



print(sys.argv)     #  sys.argv should be run in terminal to retrieve all files in the directory.



############################################  Visualization #################################


# e.g.  stacked_images = stackImages(0.5, ([1,2,3]))  :  show horizantaly 3 stacked images
# e.g.  stacked_images = stackImages(0.5, (   [1,2,3],[4,5,6],[7,8,9]    ))  : show 3*3 Matrix
#
# e.g.  stacked_images = stackImages(0.5, ([1,2,3]))  :  show horizantaly 3 stacked images
# e.g.  stacked_images = stackImages(0.5, (   [1,2,3],[4,5,6],[7,8,9]    ))  : show 3*3 Matrix

def stackImages(scale, imgArray):
    rows = len(imgArray)
    columns = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)  # will return  Boolean Value
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, columns):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:  imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), dtype='uint8')
        rows_list = [imageBlank] * rows
        for i in range(0, rows):
            rows_list[i] = imgArray[i]
        for i in range(0, rows):
            rows_list[i] = np.hstack(imgArray[i])
        ver = np.vstack(rows_list)





    else:

        for i in range(0, rows):
            if imgArray[i].shape[:2] == imgArray[0].shape[:2]:
                imgArray[i] = cv2.resize(imgArray[i], (0, 0), None, scale, scale)
            else:
                imgArray[i] = cv2.resize(imgArray[i], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[i].shape) == 2:
                imgArray[i] = cv2.cvtColor(imgArray[i], cv2.COLOR_GRAY2BGR)

        hor = np.hstack(imgArray)
        ver = hor

    return ver


######################################  1th MRF Smotthig ###############################33333

def postprocessing(im, unary):

	unary = np.float32(unary)
	unary = cv2.GaussianBlur(unary, (9, 9), 0)

	g = maxflow.Graph[float]()
	nodes = g.add_grid_nodes(unary.shape)

	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			v = nodes[i,j]
			g.add_tedge(v, 1-unary[i,j], unary[i,j])

	def potts_add_edge(i0, j0, i1, j1):
		v0, v1 = nodes[i0,j0], nodes[i1,j1]
		w = 0.1 * np.exp(-((im[i0,j0] - im[i1,j1])**2).sum() / 0.1)
		g.add_edge(v0, v1, w, w)



	for i in range(1,im.shape[0]-1):
		for j in range(1,im.shape[1]-1):
			potts_add_edge(i, j, i, j-1)
			potts_add_edge(i, j, i, j+1)
			potts_add_edge(i, j, i-1, j)
			potts_add_edge(i, j, i+1, j)

	g.maxflow()
	sgm = g.get_grid_segments(nodes)
	seg =np.float32(sgm)
	return seg

############################################## 2th MRF Smoothing ##########################





# look up the value of Y for the given indices
# if the indices are out of bounds, return 0
def compute_log_prob_helper(Y, i, j):
  try:
    return Y[i][j]
  except IndexError:
    return 0


def compute_log_prob(X, Y, i, j, w_e, w_s, y_val):
  result = w_e * X[i][j] * y_val

  result += w_s * y_val * compute_log_prob_helper(Y, i-1, j)
  result += w_s * y_val * compute_log_prob_helper(Y, i+1, j)
  result += w_s * y_val * compute_log_prob_helper(Y, i, j-1)
  result += w_s * y_val * compute_log_prob_helper(Y, i, j+1)
  return result


def denoise_image(X, w_e, w_s):
  m, n = np.shape(X)
  # initialize Y same as X
  Y = np.copy(X)
  # optimization
  max_iter = 10*m*n
  for iter in range(max_iter):
    # randomly pick a location
    i = np.random.randint(m)
    j = np.random.randint(n)
    # compute the log probabilities of both values of Y_ij
    log_p_neg = compute_log_prob(X, Y, i, j, w_e, w_s, -1)
    log_p_pos = compute_log_prob(X, Y, i, j, w_e, w_s, 1)
    # assign Y_ij to the value with higher log probability
    if log_p_neg > log_p_pos:
      Y[i][j] = -1
    else:
      Y[i][j] = 1
    if iter % 100000 == 0:
      print ('Completed', iter, 'iterations out of', max_iter)
  return Y


# preprocessing step
def read_image_and_binarize(image_file):
  # im = Image.open(image_file).convert("L")
  A = np.asarray(image_file).astype(int)
  # print(A.shape)
  A.flags.writeable = True

  A[A<128] = -1
  A[A>=128] = 1

  # print('shape{},type{}'.format(A.shape, type(A)))

  return A





def convert_from_matrix_and_save(M, filename, display=False):
  M[M==-1] = 0
  M[M==1] = 255
  im = Image.fromarray(np.uint8(M))
  if display:
    im.show()
  im.save(filename)

def get_mismatched_percentage(orig_image, denoised_image):
  diff = abs(orig_image - denoised_image) / 2
  return (100.0 * np.sum(diff)) / np.size(orig_image)



#########################################################################################
# def empty(a):
#     pass
#
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 640, 240)
#
#
# # Hue, Saturation, Value
#
# cv2.createTrackbar("Hue Min", "TrackBars", 0,179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 179,179, empty)
# cv2.createTrackbar("Sat Min", "TrackBars", 0,255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 255,255, empty)
# cv2.createTrackbar("Val Min", "TrackBars", 0,255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 255,255, empty)
#
# path ='Forging Press.jpg'
#
# while True:
#     img = cv2.imread(path)
#     img = cv2.resize(img, (500,400))
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
#     v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
#     v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
#     print(h_min, h_max, s_min, s_max, v_min, v_max)
#     lower = np.array([h_min, s_min, v_min])
#     upper = np.array([h_max, s_max, v_max])
#     mask  = cv2.inRange(imgHSV, lower, upper)
#     imgResult = cv2.bitwise_and(img, img, mask = mask)
#
#
#
#     # cv2.imshow("Original", img)
#     # cv2.imshow("HSV", imgHSV)
#     # cv2.imshow("Mask", mask )
#     # cv2.imshow("Result", imgResult)
#
#     imgstack = stackImages(0.6, ([img, imgHSV], [mask, imgResult]) )
#     cv2.imshow("Stacked Images", imgstack)
#
#
#     cv2.waitKey(1)   # Just Showing the Image 1 milisecond and goes to next loop and   *** also for the loop performance ***
#
#     List_vor = []
####################################################################################
Hue_min = 0
Hue_max = 179
Sat_min = 101
Sat_max = 225
Val_min = 211
Val_max = 255

path = 'Forging Press.jpg'

img = cv2.imread(path)

img = cv2.resize(img, (640, 480)) # img is 3D        For Post Processing is used

# img = cv2.resize(img, (500, 400))


imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([Hue_min, Sat_min, Val_min])
upper = np.array([Hue_max, Sat_max, Val_max])
mask = cv2.inRange(imgHSV, lower, upper)
imgResult = cv2.bitwise_and(img, img, mask=mask)
print(mask.shape)


mask2 = postprocessing(np.float32(img) / 255.0,  np.float32(mask) / 255.0)
mask2 = mask.astype(np.uint8)

# cv2.imwrite('mask.jpg', mask2)


# read input and arguments
orig_image = read_image_and_binarize(mask2)


# incas of using sys.argv (run it please in terminal     # if __name__ == '__main__':  ...)
# if len(sys.argv) > 2:
#   try:
#     w_e = eval(sys.argv[2])
#     w_s = eval(sys.argv[3])
#   except:
#     print ('Run as: \npython denoise.py <input_image>\npython denoise.py <input_image> <w_e> <w_s>')
#     sys.exit()
# else:

w_e = 8
w_s = 10

# add noise
# noisy_image = add_noise(orig_image)

# use ICM for denoising
denoised_image = denoise_image(orig_image, w_e, w_s)

# print the percentage of mismatched pixels
print ('Percentage of mismatched pixels: ', get_mismatched_percentage(orig_image, denoised_image))


convert_from_matrix_and_save(orig_image, 'orig_image.png', display=False)
# convert_from_matrix_and_save(noisy_image, 'noisy_image.png', display=False)
convert_from_matrix_and_save(denoised_image, 'denoised_image.png', display=False)


img1_copy = img.copy()


# FindContours supports only CV_8UC1 images when mode != CV_RETR_FLOODFILL otherwise supports CV_32SC1 images only in function 'cvStartFindContours_Impl'
denoised_image = denoised_image.astype(np.uint8)
print('denoised_image shape {} and type{} and element type{}'.format(denoised_image.shape, type(denoised_image), denoised_image.dtype))



contours, hierarchy = cv2.findContours(denoised_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
fgmask_contours = cv2.drawContours(img1_copy, contours,-1, (0,255,0), -3)
contours = sorted(contours, key=cv2.contourArea, reverse=True)  # sorting the  contours based on the values

for i in range(0, len(contours)):  # changing  3D numpy array to 2D numpy array
    contours[i] = contours[i].reshape([len(contours[i]), len(contours[i][0]) * len(contours[i][0][0])])

contours_tuple = tuple(contours)
total_contours = np.concatenate(contours_tuple, axis=0)  # concatenating all 2D contours in one 2D contours

max_col_point_index = (np.argmax(total_contours, axis=0))[0]
max_row_point_index = (np.argmax(total_contours, axis=0))[1]
min_row_point_index = (np.argmin(total_contours, axis=0))[1]
min_col_point_index = (np.argmin(total_contours, axis=0))[0]


x, y, w, h = cv2.boundingRect(total_contours)
rectangle = cv2.rectangle(img1_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)


# Plotting the image in stackmode
imgstack = stackImages(0.6, ([img, imgResult, mask], [mask2, denoised_image, rectangle]) )
cv2.imwrite('Image Collection.jpg', imgstack)
cv2.imshow("Stacked Images", imgstack)
cv2.waitKey(0)





