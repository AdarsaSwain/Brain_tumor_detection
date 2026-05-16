import numpy as np
import cv2 as cv


class DisplayDisease:
    """
    Performs Watershed-based tumor segmentation on a brain MRI image.

    Fixed issues (original code):
    1. readImage() used cv.COLOR_RGB2GRAY on a BGR ndarray from cv2.imdecode
       → changed to cv.COLOR_BGR2GRAY (correct for OpenCV-decoded images).
    2. displayDisease() used cv.COLOR_HSV2BGR at the end, treating the
       watershed result as if it were HSV – it isn't. Removed bad conversion
       and kept the RGB image so red boundary lines are visible.
    3. Watershed requires a *3-channel uint8* 'original' image. Added a guard
       to ensure self.Img is always BGR uint8 before watershed.
    """

    curImg = None
    Img = None
    thresh = None
    kernel = None
    ret = None

    def readImage(self, img: np.ndarray):
        """Accept a BGR uint8 numpy array as returned by cv2.imdecode."""
        if not isinstance(img, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        # Ensure 3-channel BGR uint8 for watershed compatibility
        if img.ndim == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        self.Img = img.copy()
        self.curImg = img.copy()

        # FIX: use COLOR_BGR2GRAY (not COLOR_RGB2GRAY) because cv2.imdecode gives BGR
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.ret, self.thresh = cv.threshold(
            gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )

    def getImage(self):
        return self.curImg

    # ── noise removal ────────────────────────────────────────────────────────
    def removeNoise(self):
        if self.thresh is None:
            raise ValueError("Input image not initialized – call readImage() first")
        self.kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(
            self.thresh, cv.MORPH_OPEN, self.kernel, iterations=2
        )
        self.curImg = opening

    # ── watershed segmentation ───────────────────────────────────────────────
    def displayDisease(self):
        if self.curImg is None:
            raise ValueError("Input image not initialized – call readImage() first")

        # sure background
        sure_bg = cv.dilate(self.curImg, self.kernel, iterations=3)

        # sure foreground via distance transform
        dist_transform = cv.distanceTransform(self.curImg, cv.DIST_L2, 5)
        _, sure_fg = cv.threshold(
            dist_transform, 0.7 * dist_transform.max(), 255, 0
        )

        # unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # marker labelling
        _, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Watershed needs the *original* BGR uint8 image
        markers = cv.watershed(self.Img, markers)

        # Mark boundaries in red
        result = self.Img.copy()
        result[markers == -1] = [0, 0, 255]   # BGR red boundary

        # FIX: Do NOT convert with COLOR_HSV2BGR – result is already BGR
        self.curImg = result

    # ── calculate tumor area percentage ──────────────────────────────────────
    def calculateTumorPercentage(self):
        if self.Img is None:
            raise ValueError("Input image not initialized – call readImage() first")
        gray = cv.cvtColor(self.curImg, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        total_pixels = binary.shape[0] * binary.shape[1]
        nonzero = cv.countNonZero(binary)
        percentage = round((nonzero / total_pixels) * 100, 2)
        return percentage
