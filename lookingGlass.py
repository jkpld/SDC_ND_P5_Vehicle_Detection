import numpy as np
import cv2
from collections import deque
from scipy.ndimage.measurements import label

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# LookingGlass will store a history of the predicted location of objects.
# It has functionality to add new predictions and to then locate the
# objects based on the history of predictions
class LookingGlass():
    def __init__(self, history_size=5, threshold=10):
        self.looking_glass = deque(maxlen=history_size)
        self.threshold = threshold
        self.looking_glass_size = None

    # Add lookin_glass history_size getter/setter
    @property
    def history_size(self):
        return self.looking_glass.maxlen

    @history_size.setter
    def history_size(self, new_size):
        self.looking_glass = deque(self.looking_glass, maxlen=new_size)

    def add_new_frame(self, bboxes, size):
        if self.looking_glass_size is None:
            self.looking_glass_size = size

        else:
            if self.looking_glass_size != size:
                print('Warning! Looking glass sizes to not match, using initial size.')
                size = self.looking_glass_size

        lg = np.zeros(size,dtype=np.uint8)
        for box in range(bboxes.shape[0]):
            center = (bboxes[box,0:2] + bboxes[box,2:4])/2
            lg[int(center[1]-30):int(center[1]+30),int(center[0]-30):int(center[0]+30)] += 1

        self.looking_glass.append(lg)

    def locate_objects(self, return_lookingGlass=False, min_area=None, threshold=None):
        if self.looking_glass_size is None:
            print('Looking glass is empty! No objects to be found.')
            return None

        glass = np.zeros_like(self.looking_glass[-1])
        for lg in self.looking_glass:
            glass += lg

        if threshold is None:
            th = self.threshold
        else:
            th = threshold

        objects_bw = (glass >= th).astype(np.uint8)

        # http://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
        output = cv2.connectedComponentsWithStats(objects_bw, 4, cv2.CV_32S)

        # Object bounding boxes
        bboxes = output[2][1:,:4]
        bboxes[:,2:4] += bboxes[:,0:2]
        # Object centroids
        centroids = output[3][1:,:]
        # Areas
        areas = output[2][1:,4]

        if min_area is not None:
            keep = areas >= min_area
            bboxes = bboxes[keep,:]
            centroids = centroids[keep,:]
            areas = areas[keep]

        # Store is list of dictionaries
        objects = {'num_objects': output[0]-1,
                    'bboxes': bboxes,
                    'centroids': centroids,
                    'areas': areas}

        if return_lookingGlass:
            return objects, glass
        else:
            return objects

    def reset(self):
        self.looking_glass.clear()
        self.looking_glass_size = None
