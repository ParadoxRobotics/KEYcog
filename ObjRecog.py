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
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.resize(img, (640,480))
# compute Hue and Saturation
hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
H, S, V = cv2.split(hsvImg)
# compute gradient
dx = cv2.Sobel(imgGray, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(imgGray, cv2.CV_32F, 0, 1)
# convert to float
dx = dx.astype(float)
dy = dy.astype(float)
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


plt.imshow(imgGray)
plt.show()
plt.imshow(H)
plt.show()
plt.imshow(S)
plt.show()
plt.imshow(mag)
plt.show()
plt.imshow(angle)
plt.show()

# merge the different channel into a unified matrix F = [H, S, Mag, Ang]
F = cv2.merge((H, S, mag, angle))
