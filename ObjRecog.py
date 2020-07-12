# Robust, efficint and simple single objet recognition for robot application
# use LSD python wrapper : https://github.com/xanxys/lsd-python
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
"""

import numpy as np
import cv2
import lsd
from matplotlib import pyplot as plt
from matplotlib import pyplot

# create object model
class targetModel():
    def __init__(self, ):
        # detector class -> only for drawing
        self.fld = cv2.ximgproc.createFastLineDetector()
        # base input
        self.imageTarget = None
        self.maskTarget = None
        # model
        self.LSDLine = None
        self.descriptor = None
        self.position = None

    def createModel(self, img, mask):
        # store input data
        self.imageTarget = img
        if mask is not None:
            self.maskTarget = mask
        # compute line
        imgLine_ = lsd.detect_line_segments(self.imageTarget.astype('float64'), scale=0.8, sigma_scale=0.6, quant=2.0, ang_th=22.5, log_eps=0.0, density_th=0.7, n_bins=1024)
        self.LSDLine = np.reshape(imgLine_[:,0:4], (imgLine_.shape[0],1,4)).astype('float32')
        # plot
        plt.imshow(self.fld.drawSegments(self.imageTarget, self.LSDLine), interpolation='nearest', aspect='auto')
        plt.show()

img = cv2.imread('target.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (640,480))
model = targetModel()
model.createModel(img, None)
