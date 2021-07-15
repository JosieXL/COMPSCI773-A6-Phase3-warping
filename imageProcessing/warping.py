#!/usr/bin/env python

# utilities.py - Utility functions for image processing based on pixel arrays
#
# Copyright (C) 2020 Martin Urschler <martin.urschler@auckland.ac.nz>
#
# Original concept by Martin Urschler.
#
# LICENCE (MIT)
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import numpy as np

def warpingPerspectiveForward(left_img, right_img, image_width, image_height, homography, blend_factor = 0.5):

    # createInitializedGreyscalePixelArray
    warped = [[0 for x in range(2 * image_width)] for y in range(image_height)]

    # with u and v we go over the pixel grid of the output warped image, which is twice the size of the left image
    for v in range(image_height):
        for u in range(2 * image_width):
            # transform u,v to point p in homogeneous coordinates (stored as numpy array)
            p_hom = np.asarray([u, v, 1])
            p_transformed_hom = np.dot(homography, p_hom)
            # transform back from homogoneous coordinates to a point "x, y" in R^2
            # x, y is the point in the left image, falling between 4 pixels
            x = p_transformed_hom[0] / p_transformed_hom[2]
            y = p_transformed_hom[1] / p_transformed_hom[2]

            # pixel top left is the one we get after we convert floats to ints
            x1 = math.floor(x)
            y1 = math.floor(y)

            value_from_right_img = 0.0
            value_from_right_img_computed = False
            # need to check if that one is outside the bounds of the right image
            if x1 >= 0 and x1 < image_width-1 and y1 >= 0 and y1 < image_height-1:
                a = x-x1
                b = y-y1

                value_from_right_img_computed = True
                value_from_right_img += (1.0 - a) * (1.0 - b) * right_img[y1][x1]
                value_from_right_img += a * b * right_img[y1+1][x1+1]
                value_from_right_img += (1.0 - a) * b * right_img[y1+1][x1]
                value_from_right_img += a * (1.0 - b) * right_img[y1][x1+1]

            # if we are over a pixel of the left image, we need to blend
            if u <= image_width - 1:
                value_from_left_img = left_img[v][u]
                if value_from_right_img_computed:
                    warped[v][u] = blend_factor * value_from_left_img + (1.0 - blend_factor) * value_from_right_img
                else:
                    warped[v][u] = value_from_left_img
            else:
                warped[v][u] = value_from_right_img

    return warped