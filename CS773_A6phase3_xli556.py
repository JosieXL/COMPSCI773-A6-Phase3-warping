from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch

from timeit import default_timer as timer

from numpy.lib.shape_base import array_split

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth
import imageProcessing.warping as IPWarp

import numpy as np
import random


# this is a helper function that puts together an RGB image for display in matplotlib, given
# three color channels for r, g, and b, respectively
def prepareRGBImageFromIndividualArrays(r_pixel_array,g_pixel_array,b_pixel_array,image_width,image_height):
    rgbImage = []
    for y in range(image_height):
        row = []
        for x in range(image_width):
            triple = []
            triple.append(r_pixel_array[y][x])
            triple.append(g_pixel_array[y][x])
            triple.append(b_pixel_array[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage


# takes two images (of the same pixel size!) as input and returns a combined image of double the image width
def prepareMatchingImage(left_pixel_array, right_pixel_array, image_width, image_height):

    matchingImage = IPUtils.createInitializedGreyscalePixelArray(image_width * 2, image_height)
    for y in range(image_height):
        for x in range(image_width):
            matchingImage[y][x] = left_pixel_array[y][x]
            matchingImage[y][image_width + x] = right_pixel_array[y][x]

    return matchingImage

# this following two functions are used to compute the sobel filter
def computeVerticalEdgesSobel(pixel_array, image_width, image_height):
    Matrix = np.ones((image_height,image_width))
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            Matrix[0][j] = 0.000
            Matrix[-1][j] = 0.000
            Matrix[i][0] = 0.000
            Matrix[i][-1] = 0.000
            
            if (Matrix[i][j]==1.000):
                Matrix[i][j] = (-1)*pixel_array[i-1][j-1] + (-2)*pixel_array[i][j-1] + (-1)*pixel_array[i+1][j-1] + 0*pixel_array[i-1][j] + 0*pixel_array[i][j] + 0*pixel_array[i+1][j] + (1)*pixel_array[i+1][j+1] + (2)*pixel_array[i][j+1] + (1)*pixel_array[i-1][j+1]
    return Matrix

def computeHorizontalEdgesSobel(pixel_array, image_width, image_height):
    Matrix = np.ones((image_height,image_width))
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            Matrix[0][j] = 0.000
            Matrix[-1][j] = 0.000
            Matrix[i][0] = 0.000
            Matrix[i][-1] = 0.000
            
            if (Matrix[i][j]==1.000):
                Matrix[i][j] = (-1)*pixel_array[i+1][j-1] + (-2)*pixel_array[i+1][j] + (-1)*pixel_array[i+1][j+1] + (0)*pixel_array[i][j-1] + (0)*pixel_array[i][j] + (0)*pixel_array[i][j+1] + (1)*pixel_array[i-1][j-1] + (2)*pixel_array[i-1][j] + (1)*pixel_array[i-1][j+1]
    return Matrix

# function for sobel filter and outputs Ix2, Iy2 and IxIy
def SobelDerivativeFilter(pixel_array, image_width, image_height):
    
    # sobel kernel
    #horizontal_kernel_Iy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    #vertical_kernel_Ix = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    Ix = computeVerticalEdgesSobel(pixel_array, image_width, image_height)
    Iy = computeHorizontalEdgesSobel(pixel_array, image_width, image_height)
    
    # if type is np.matrix, then A*B means matrix multiplication
    # if type is np.ndarray, then A*B means Hadamard product
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    IxIy = Ix * Iy 

    #type of Ix2, Iy2 and IxIy are <class 'numpy.ndarray'>
    return Ix2, Iy2, IxIy

# take a kernel_size and a sigma to create the gaussian kernel
def create_gauss_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size//2
    if sigma == 0:
        sigma = ((kernel_size - 1)*0.5-1)*0.3+0.8
    for i in range(kernel_size):
        for j in range(kernel_size):
            x,y = i-center, j-center
            kernel[i,j] = 1/(2*np.pi*sigma**2) * np.exp(-(x**2+y**2)/(2*(sigma**2)))         
    kernel = kernel/np.sum(kernel)
    return kernel

# function for calculate Gaussian Filter
def GaussianFilter(pixel_array, image_width, image_height, kernel_size, sigma):
    a = int(0.5*(kernel_size-1))
    kernel = create_gauss_kernel(kernel_size,sigma)
    gaussian = np.zeros((image_height,image_width))
    paddingArr = np.pad(pixel_array, pad_width=a, mode='constant', constant_values=0.0)
    for i in range(len(gaussian)):
        for j in range(len(gaussian[i])):
            gaussian[i][j] = np.sum(paddingArr[i:(i+kernel_size), j:(j+kernel_size)]* kernel[:,None])
    return gaussian


# if a matrix is a squared matrix, then can apply following implementation of cornerness score function C
# takes a square n*n matrix M, calculate its determinant
#def det(M):
#    return np.linalg.det(M)
# takes a square n*n matrix M, calculate its trace
#def trace(M):
#    return np.trace(M)
# Cornerness score function C = det(M) - a*trace(M)^2 where Harris constant a=0.04 here
#def CornernessScore(M, a):
#    C = det(M) - a*trace(M)*trace(M)
#return C

# if a matrix is not a squared matrix, then do following with g(Ix2) = gx, g(Iy2) = gy and g(IxIy) = gxy
def CornernessScore(gx, gy, gxy, a=0.04):
    C = gx * gy - gxy**2 - a*((gx + gy)**2)
    return C

# this function is to check if a corner is negative, if it is then changed to 0.0, otherwise keeps the same
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    thresholded = np.array(IPUtils.createInitializedGreyscalePixelArray(image_width, image_height), dtype=np.int64)
    for i in range(len(thresholded)):
        for j in range(len(thresholded[i])):
            if pixel_array[i][j] < threshold_value:
                thresholded[i][j] = 0.0
            else:
                thresholded[i][j] = pixel_array[i][j]
    return thresholded

# function to compute the first 1000 strongest corner and store the coordinates as tuples inside a list
def computeFirst1000StrongestCornerTupleList(pixel_array, image_width, image_height):
    # As append into a numpy.ndarray is much slower than append into a list, so I create an empty list here
    Corner_tuple_list = list()
    pre = np.pad(pixel_array, pad_width=1, mode='constant', constant_values=0.00)
    for i in range(len(pixel_array)):
        for j in range(len(pixel_array[i])):
            if (pixel_array[i][j] > pre[i][j]) and (pixel_array[i][j] > pre[i+1][j]) and (pixel_array[i][j] > pre[i+2][j]) and (pixel_array[i][j] > pre[i][j+1]) and (pixel_array[i][j] > pre[i+2][j+1]) and (pixel_array[i][j] > pre[i+2][j+2]) and (pixel_array[i][j] > pre[i+1][j+2]) and (pixel_array[i][j] > pre[i][j+2]):
                Corner_tuple_list.append((j, i, pixel_array[i][j]))
    Corner_tuple_list.sort(key=lambda tup: tup[2], reverse=True)
    one_thousand_Corner_tuple_list = Corner_tuple_list[:1000]
    oneThousandStrongestCorner = list(dict.fromkeys(one_thousand_Corner_tuple_list))
    return oneThousandStrongestCorner

# Compute the NCC for 15*15 block from left image with 15*15 block from right image where the mean coordinator from two blocks are the Harris Corners
def NormalizeCrossCorrelation(arrL, arrR):
    NCC = np.sum((arrL-np.mean(arrL))*(arrR-np.mean(arrR)))/((np.sum((arrL-np.mean(arrL))**2)**0.5)*(np.sum((arrR-np.mean(arrR))**2)**0.5))
    return NCC

def ratioOfBestAndSecondMatching(bestMatch, secondBestMatch):
	return secondBestMatch/bestMatch

def FeatureMatching(px_array_left, px_array_right, oneThousandCornerL, oneThousandCornerR, window_size):
    aw = int(0.5*(window_size-1))
    paddingArrL = np.pad(px_array_left, pad_width=aw, mode='constant', constant_values=0.0)
    paddingArrR = np.pad(px_array_right, pad_width=aw, mode='constant', constant_values=0.0)
    match_list = list()
    for i in range(len(oneThousandCornerL)):
        NCC_list = list()
        cornerXL, cornerYL = oneThousandCornerL[i][0], oneThousandCornerL[i][1]
        blockL = paddingArrL[cornerYL:(cornerYL+window_size),cornerXL:(cornerXL+window_size)]
        for j in range(len(oneThousandCornerR)):
            cornerXR, cornerYR = oneThousandCornerR[j][0], oneThousandCornerR[j][1]
            blockR = paddingArrR[cornerYR:(cornerYR+window_size),cornerXR:(cornerXR+window_size)]

            # numpy.flatten() is to convert a 2d array to 1d array        
            NCC = NormalizeCrossCorrelation(blockL.flatten(), blockR.flatten())
            if len(NCC_list) < 2:
                NCC_list.append(((cornerXL, cornerYL), (cornerXR, cornerYR), NCC))
                NCC_list = list(dict.fromkeys(NCC_list))
                NCC_list.sort(key=lambda tup: tup[2],reverse=True)
            elif len(NCC_list) == 2:
                bestMatch, secondBestMatch = NCC_list[0][2], NCC_list[1][2]
                if NCC > bestMatch:
                    NCC_list[1] = NCC_list[0]
                    NCC_list[0] = ((cornerXL, cornerYL), (cornerXR, cornerYR), NCC)
                elif bestMatch > NCC > secondBestMatch:
                    NCC_list[1] = ((cornerXL, cornerYL), (cornerXR, cornerYR), NCC)
        bestMatch1, secondBestMatch1 = NCC_list[0][2], NCC_list[1][2]
        #Heuristic - ratio of bestMatch to secondMatch should be < 0.9
        if ratioOfBestAndSecondMatching(bestMatch1, secondBestMatch1) < 0.9:
            match_list.append(NCC_list[0])
    return match_list

# Direct linear transformation algorithm with parameter 
# matching_array = np.array([((x1,y1),(x1',y1')),((x2,y2),(x2',y2')),((x3,y3),(x3',y3')),((x4,y4),(x4',y4')),...]) 
# len(matching_array) >= 4
def DLT(matching_array):
    matching_array = np.array(matching_array)
    A = np.zeros((2*len(matching_array),9))
    for i in range(0,len(A),2):
        index = np.int(i/2)
        x, y = matching_array[index][0][0], matching_array[index][0][1]
        u, v = matching_array[index][1][0], matching_array[index][1][1]
        A[i] = [0, 0, 0, x, y, 1, -x*v, -y*v, -v]
        A[i+1] = [x, y, 1, 0, 0, 0, -x*u, -y*u, -u]
    # Perform Singular Value Decomposition
    [U, S, Vt] = np.linalg.svd(A)
    # last row of Vt is the smallest singular value
    nonhomogenous_H = Vt[-1]
    # normalize it by dividing the last element hence H33=1 and reshape to 3x3
    H = (nonhomogenous_H/nonhomogenous_H[-1]).reshape((3,3))
    return H

def pointsAreCollinear(p1,p2,p3):
	areaOfTriangle = 0.5 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
	return areaOfTriangle < 1e-5

# MatchingList is [(p1 in left image, p1_match in right image), (p2 in left image, p2_match in right image), ...]
# inlierMappingThreshold = 1-2 pixels
# numberOfRandomDraws > 1000
def RANSACLoop(MatchingList, inlierMappingThreshold, numberOfRandomDraws):
    InlinerList = []
    for x in range(numberOfRandomDraws):
        setOfInlinerMatching = []
        NumberOfInliners = 0
        [(p1,p1m), (p2,p2m), (p3, p3m), (p4, p4m)] = random.sample(MatchingList, 4)
        while( (pointsAreCollinear(p1,p2,p3) == True) or (pointsAreCollinear(p1,p2,p4) == True) or (pointsAreCollinear(p1,p3,p4) == True) or (pointsAreCollinear(p2,p3,p4) == True) or (pointsAreCollinear(p1m,p2m,p3m) == True) or (pointsAreCollinear(p1m,p2m,p4m) == True) or (pointsAreCollinear(p1m,p3m,p4m) == True) or (pointsAreCollinear(p2m,p3m,p4m) == True) ):
            [(p1,p1m), (p2,p2m), (p3, p3m), (p4, p4m)] = random.sample(MatchingList, 4)
        four_matching_array = [(p1,p1m), (p2,p2m), (p3, p3m), (p4, p4m)]
        H = DLT(four_matching_array)
        for i in range(len(MatchingList)):
            homogeneous_checkpoint = np.array([MatchingList[i][0][0], MatchingList[i][0][1], 1])
            homogeneous_checkpoint_vertical = np.vstack(homogeneous_checkpoint)
            nonhomogeneous_returnpoint = np.dot(H, homogeneous_checkpoint_vertical)
            homogeneous_returnpoint_horizontal = np.hstack(nonhomogeneous_returnpoint/nonhomogeneous_returnpoint[-1])
            [x_1, y_1] = homogeneous_returnpoint_horizontal[0:2]
            [x1m, y1m] = [MatchingList[i][1][0], MatchingList[i][1][1]]
            distance = np.sqrt( (x_1-x1m)**2 + (y_1-y1m)**2 )
            if distance < inlierMappingThreshold:
                NumberOfInliners += 1
                setOfInlinerMatching.append(((MatchingList[i][0][0], MatchingList[i][0][1]), (MatchingList[i][1][0], MatchingList[i][1][1])))
        if len(InlinerList) == 0:
            InlinerList.append([H, NumberOfInliners, setOfInlinerMatching])
        else:
            if NumberOfInliners > InlinerList[0][1]:
                InlinerList[0] = [H, NumberOfInliners, setOfInlinerMatching]
    return InlinerList[0]

# This is our code skeleton that performs the stitching
def main():
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"
    
    (image_width, image_height, px_array_left_original)  = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_left_image)
    (image_width, image_height, px_array_right_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_right_image)
    
    start = timer()
    px_array_left = IPSmooth.computeGaussianAveraging3x3(px_array_left_original, image_width, image_height)
    px_array_right = IPSmooth.computeGaussianAveraging3x3(px_array_right_original, image_width, image_height)
    end = timer()
    print("elapsed time image smoothing: ", end - start)

    # make sure greyscale image is stretched to full 8 bit intensity range of 0 to 255
    px_array_left = IPPixelOps.scaleTo0And255AndQuantize(px_array_left, image_width, image_height)
    px_array_right = IPPixelOps.scaleTo0And255AndQuantize(px_array_right, image_width, image_height)
    
    # type of px_array_left and px_array_right are <class 'list'>
    # convert them into numpy.array to make it faster
    px_array_left = np.array(px_array_left)
    px_array_right = np.array(px_array_right)

    # the variable which end with 'L' is for px_array_left while end with 'R' is for px_array_right

    Ix2L, Iy2L, IxIyL = SobelDerivativeFilter(px_array_left, image_width, image_height)
    Ix2R, Iy2R, IxIyR = SobelDerivativeFilter(px_array_right, image_width, image_height)

    kernel_size = 5 # 5x5 Gaussian Window - must be odd number
    sigma = 1 # can be calculated by sigma = 0.25*(kernel_size-1)
    
    gxL = GaussianFilter(Ix2L, image_width, image_height, kernel_size, sigma)
    gyL = GaussianFilter(Iy2L, image_width, image_height, kernel_size, sigma)
    gxyL = GaussianFilter(IxIyL, image_width, image_height, kernel_size, sigma)

    gxR = GaussianFilter(Ix2R, image_width, image_height, kernel_size, sigma)
    gyR = GaussianFilter(Iy2R, image_width, image_height, kernel_size, sigma)
    gxyR = GaussianFilter(IxIyR, image_width, image_height, kernel_size, sigma)

    C_arrL = CornernessScore(gxL, gyL, gxyL, 0.04)
    C_arrR = CornernessScore(gxR, gyR, gxyR, 0.04)

    C_after_thresholdL = computeThresholdGE(C_arrL, 0.0, image_width, image_height)
    C_after_thresholdR = computeThresholdGE(C_arrR, 0.0, image_width, image_height)
    
    oneThousandStrongestCornerL = computeFirst1000StrongestCornerTupleList(C_after_thresholdL, image_width, image_height)
    oneThousandStrongestCornerR = computeFirst1000StrongestCornerTupleList(C_after_thresholdR, image_width, image_height)
    

    # some visualizations
    '''
    # visualization for A6 Phase 1
    fig1, axs1 = pyplot.subplots(1, 2)

    axs1[0].set_title('Harris response left overlaid on orig image')
    axs1[1].set_title('Harris response right overlaid on orig image')
    axs1[0].imshow(px_array_left, cmap='gray')
    axs1[1].imshow(px_array_right, cmap='gray')

    
    # plot a red point in the center of each image
    for i in range(len(oneThousandStrongestCornerL)):
        circleL = Circle((oneThousandStrongestCornerL[i][0], oneThousandStrongestCornerL[i][1]), 1, color='r')
        axs1[0].add_patch(circleL)
    
    for i in range(len(oneThousandStrongestCornerR)):
        circleR = Circle((oneThousandStrongestCornerR[i][0], oneThousandStrongestCornerR[i][1]), 1, color='r')
        axs1[1].add_patch(circleR)

    pyplot.show()
    '''

    oneThousandCornerL = np.array(oneThousandStrongestCornerL)
    oneThousandCornerR = np.array(oneThousandStrongestCornerR)

    # window_size - 15x15, can be changed
    matching_list = FeatureMatching(px_array_left, px_array_right, oneThousandCornerL, oneThousandCornerR, 15)
    match_arr = np.array(matching_list)

    # a combined image including a red matching line as a connection patch artist (from matplotlib\)
    # visualization for A6 Phase 2
    '''
    matchingImage = prepareMatchingImage(px_array_left, px_array_right, image_width, image_height)
    
    pyplot.imshow(matchingImage, cmap='gray')
    ax = pyplot.gca()
    ax.set_title("Matching image")
    for i in range(len(match_arr)):
        pointA = (match_arr[i][0][0], match_arr[i][0][1])
        pointB = (match_arr[i][1][0]+image_width, match_arr[i][1][1])
        connection = ConnectionPatch(pointA, pointB, "data", edgecolor='r', linewidth=0.5)
        ax.add_artist(connection)
    pyplot.show()
    '''

    MatchingList = []
    for i in range(len(match_arr)):
        MatchingList.append(((match_arr[i][0][0],match_arr[i][0][1]),(match_arr[i][1][0],match_arr[i][1][1])))
    MatchingList = list(dict.fromkeys(MatchingList))
    [H, highest_NumberOfInliners, setOfInlinerMatching] = RANSACLoop(MatchingList,1,1001)
    # recompute DLT on all inliers
    H_best = DLT(setOfInlinerMatching)

    warp = IPWarp.warpingPerspectiveForward(px_array_left, px_array_right, image_width, image_height, H_best, blend_factor = 0.5)

    pyplot.imshow(warp, cmap='gray')
    pyplot.title("Warped image of current homography")
    pyplot.show()


if __name__ == "__main__":
    main()

