import numpy as np
import cv2

class Frame():

    color_conversion_table = {'RGB2BGR': cv2.COLOR_RGB2BGR,
                              'BGR2RGB': cv2.COLOR_RGB2BGR,
                              'RGB2LUV': cv2.COLOR_RGB2LUV,
                              'RGB2LAB': cv2.COLOR_RGB2LAB,
                              'RGB2HLS': cv2.COLOR_RGB2HLS,
                              'RGB2YCrCb': cv2.COLOR_RGB2YCrCb,
                              'RGB2HSV': cv2.COLOR_RGB2HSV,
                              'BGR2LUV': cv2.COLOR_BGR2LUV,
                              'BGR2LAB': cv2.COLOR_BGR2LAB,
                              'BGR2HLS': cv2.COLOR_BGR2HLS,
                              'BGR2YCrCb': cv2.COLOR_BGR2YCrCb,
                              'BGR2HSV': cv2.COLOR_BGR2HSV}

    def __init__(self, img, colorspace='RGB'):
        self.image = {'RGB': None, 'BGR': None, 'LUV': None, 'HLS': None,
            'YUV': None, 'YCrCb': None, 'LAB': None, 'HSV': None}
        self.input_color = colorspace

        if type(img)==np.ndarray:
            if np.all(img<=1):
                self.image[colorspace] = np.uint8(np.copy(img)*255)
            else:
                self.image[colorspace] = np.copy(img)
        else:
            img = cv2.imread(img)
            colorspace = 'BGR'
            self.input_color = colorspace
            self.image['BGR'] = img

    def _get_color(self, col_out):
        if self.image[col_out] is None:
            conversion_code = self.input_color+'2'+col_out
            if conversion_code in self.color_conversion_table:
                self.image[col_out] = cv2.cvtColor(self.image[self.input_color], self.color_conversion_table[conversion_code])
            else:
                print('Warning! Color conversion not found in color_conversion_table.')

    @property
    def RGB(self):
        self._get_color('RGB')
        return np.copy(self.image['RGB'])
    @property
    def BGR(self):
        self._get_color('BGR')
        return np.copy(self.image['BGR'])
    @property
    def LUV(self):
        self._get_color('LUV')
        return np.copy(self.image['LUV'])
    @property
    def LAB(self):
        self._get_color('LAB')
        return np.copy(self.image['LAB'])
    @property
    def HLS(self):
        self._get_color('HLS')
        return np.copy(self.image['HLS'])
    @property
    def HSV(self):
        self._get_color('HSV')
        return np.copy(self.image['HSV'])
    @property
    def YCrCb(self):
        self._get_color('YCrCb')
        return np.copy(self.image['YCrCb'])
    @property
    def YUV(self):
        self._get_color('YUV')
        return np.copy(self.image['YUV'])
