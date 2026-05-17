import cv2 as cv
import numpy as np

class DisplayTumor:
    def __init__(self):
        self.Img = None
        self.curImg = None
        self.thresh = None
        self.kernel = None
        self.tumorPercentage = None

    def readImage(self, img):
        if isinstance(img, np.ndarray):
            self.Img = np.array(img)
            self.curImg = np.array(img)
            gray = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
            self.ret, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        else:
            raise ValueError("Input image must be a numpy array")

    def getImage(self):
        return self.curImg

    def removeNoise(self):
        if self.thresh is not None:
            self.kernel = np.ones((3, 3), np.uint8)
            opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, self.kernel, iterations=2)
            self.curImg = opening
        else:
            raise ValueError("Input image not initialized")

    def displayTumor(self):
        if self.curImg is not None:
            # sure background area
            sure_bg = cv.dilate(self.curImg, self.kernel, iterations=3)

            # Finding sure foreground area
            dist_transform = cv.distanceTransform(self.curImg, cv.DIST_L2, 5)
            ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

            # Find unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1

            # Now mark the region of unknown with zero
            markers[unknown == 255] = 0
            markers = cv.watershed(self.Img, markers)
            self.Img[markers == -1] = [255, 0, 0]

            tumorImage = cv.cvtColor(self.Img, cv.COLOR_HSV2BGR)
            self.curImg = tumorImage

            # Calculate percentage of tumor part in the image
            self.calculateTumorPercentage()
            return self.tumorPercentage

        else:
            raise ValueError("Input image not initialized")

    def calculateTumorPercentage(self):
     if self.Img is not None:
        gray = cv.cvtColor(self.Img, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        self.tumorPercentage = round((cv.countNonZero(binary) / (binary.shape[0] * binary.shape[1])) * 100, 2)
        return self.tumorPercentage
     else:
        raise ValueError("Input image not initialized")
