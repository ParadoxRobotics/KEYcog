# Robust, efficint and simple single objet recognition for robot application
# this work is based on covariant matrix descriptor and deformable template matching (DTM)
# Author : Munch Quentin, 2020.

"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
# Robust, efficint and simple single objet recognition for robot application
# Author : Munch Quentin, 2020.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

# get image and resize it
img = cv2.imread('target.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640,480))
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# compute Hue and Saturation
hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
H, S, V = cv2.split(hsvImg)
# compute gradient
dx = cv2.Sobel(imgGray, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(imgGray, cv2.CV_32F, 0, 1)
# compute gradient magnitude and angle (in degree)
mag = cv2.magnitude(dx, dy)
# thresholding the magnitude value
thresh = 50
ret, magMask = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)
magMask = magMask.astype(np.uint8)
# apply the mask to the magnitude
mag = cv2.bitwise_and(mag, mag, mask=magMask)
# compute the angle given the thresholding magnitude mask
angle = cv2.phase(dx, dy, angleInDegrees=True)
angle = cv2.bitwise_and(angle, angle, mask=magMask)

# merge the different channel into a unified matrix F = [H, S, Mag, Ang]
F = np.concatenate((H, S, mag, angle), axis=0)

# compute image integral
sum, sqsum, tilted	= cv2.integral3(imgGray)
