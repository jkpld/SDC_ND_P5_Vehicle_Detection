import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

from frame import Frame
from lookingGlass import LookingGlass
from tic_toc import *

from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


class CarClassifier():
    allowed_classifiers = ('SVM','AdaBoost','DecisionTree')
    default_feature_options = {'colorspace': 'LAB',
                'spatial_size': (16, 16),
                'spatial_feat': True,
                'hist_bins': 32,
                'hist_range': (0,256),
                'hist_feat': True,
                'hog_channel': 0,
                'orient': 9,
                'pix_per_cell': 8,
                'cell_per_block': 2,
                'hog_feat': True,
                'window': 64}

    search_options_helper = lambda cps, roi, scale : \
        {'cells_per_step': cps, 'roi': roi, 'scale': scale}

    default_search_options = [search_options_helper(1, [0.3,0.9,0,1], 6),
                              search_options_helper(2, [0.4,0.9,0,1], 4),
                              search_options_helper(2, [0.5,0.8,0,1], 2),
                              search_options_helper(2, [0.5,0.8,0,1], 1.5),
                              search_options_helper(2, [0.5,0.7,0.1,0.9], 1)]

    def __init__(self, classifier='SVM', classifier_options_dict=None,
                    feature_options=None, history_size=5):


        self.set_classifier(classifier, classifier_options_dict) # Note this also creats the HOG descriptor
        self.__fop = None
        self.feature_options = feature_options
        self.search_options = self.default_search_options
        self.feature_Normalizer = None
        self.features_to_use = None

        self.looking_glass = LookingGlass()

    # feature_options getter/setter
    @property
    def feature_options(self):
        return self.__fop

    @feature_options.setter
    def feature_options(self, feature_options=None):
        if self.__fop is None:
            self.__fop = self.default_feature_options

        if feature_options is not None:
            for key, value in feature_options.items():
                self.__fop[key] = value

        window = self.__fop['window']
        pix_per_cell = self.__fop['pix_per_cell']
        pix_per_block = pix_per_cell*self.__fop['cell_per_block']

        winSize = (window, window)
        cellSize = (pix_per_cell, pix_per_cell)
        blockSize = (pix_per_block, pix_per_block)
        blockStride = cellSize
        nbins = self.__fop['orient']

        self.hogDescriptor = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

    def set_classifier(self, classifier='SVM', classifier_options_dict=None):
        if classifier not in self.allowed_classifiers:
            classifer = self.allowed_classifiers[0]
            classifier_options_dict = None
            print('Warning! Unknown classifier option, using a linear SVM.')

        if classifier == 'SVM':
            if classifier_options_dict is None:
                self.classifier = LinearSVC()
            else:
                self.classifier = LinearSVC(**classifier_options_dict)
        elif classifier == 'AdaBoost':
            if classifier_options_dict is None:
                self.classifier =  AdaBoostClassifier()
            else:
                self.classifier = AdaBoostClassifier(**classifier_options_dict)
        elif classifier == 'DecisionTree':
            if classifier_options_dict is None:
                self.classifier =  DecisionTreeClassifier()
            else:
                self.classifier = DecisionTreeClassifier(**classifier_options_dict)

        self.feature_Normalizer = None
        self.features_to_use = None

    @property
    def feature_category_size(self):
        # Get number of features from each feature category (spatial,
        # color histogram, HOG)
        num_features = []

        if self.__fop['spatial_feat']:
            spatial_size = self.__fop['spatial_size']
            spatial_features = spatial_size[0]*spatial_size[1]*3
            num_features.append(spatial_features)

        if self.__fop['hist_feat']:
            hist_features = self.__fop['hist_bins']*3
            num_features.append(hist_features)

        if self.__fop['hog_feat']:
            hog_features = self.__fop['orient'] * \
                            self.__fop['cell_per_block']**2 * \
                            (64/self.__fop['pix_per_cell'] -
                            self.__fop['cell_per_block'] + 1)**2

            if self.__fop['hog_channel'] == 'ALL':
                hog_features *= 3
            num_features.append(hog_features)

        return num_features

    def train(self, X,y, use_best_of_each_category=False, use_best=False):

        # Normalize data
        self.feature_Normalizer = StandardScaler().fit(X)
        X = self.feature_Normalizer.transform(X)

        if use_best_of_each_category:
            # Run a decision tree on each category of features. Use the
            # best features from each catory for the final classifier

            feature_category = self.feature_category_size
            feature_categories = np.repeat(np.arange(len(feature_category)),feature_category)

            features_to_use = np.zeros(X.shape[1],dtype=bool)
            for cat in range(np.max(feature_categories)):
                # tree = AdaBoostClassifier(n_estimators=20)
                tree = DecisionTreeClassifier(min_samples_split=10)
                tree.fit(X[:,feature_categories==cat],y)
                features_to_use[feature_categories==cat] = tree.feature_importances_ > 1e-3

            self.classifier.fit(X[:,features_to_use],y)
            self.features_to_use = features_to_use

        elif use_best:
            # Run a decision tree on all of the features and use the best
            # for the final classifier

            # tree = AdaBoostClassifier()
            tree = DecisionTreeClassifier(min_samples_split=10)
            tree.fit(X,y)
            features_to_use = tree.feature_importances_ > 0

            self.classifier.fit(X[:,features_to_use],y)
            self.features_to_use = features_to_use
        else:
            # Use all features for the classifier
            self.classifier.fit(X, y)

    def test(self, X,y):

        if self.feature_Normalizer is None:
            print('Classifier has not yet been trained.')
            return 0
        else:
            X = self.feature_Normalizer.transform(X)
            if self.features_to_use is None:
                score =  self.classifier.score(X,y)
            else:
                score = self.classifier.score(X[:,self.features_to_use],y)
            return score

    def predict(self, X, threshold=0.515):
        if self.feature_Normalizer is None:
            print('Classifier has not yet been trained.')
            return np.zeros(X.shape[0])
        else:
            X = self.feature_Normalizer.transform(X)
            if self.features_to_use is None:
                score = self.classifier.predict_proba(X)[:,1] > threshold
            else:
                score = self.classifier.predict_proba(X[:,self.features_to_use])[:,1] > threshold
            return score

    def save_classifier(self, file_name = './car_classifier.pickle'):
        classifier = {'classifier': self.classifier,
                    'feature_options': self.__fop,
                    'feature_Normalizer': self.feature_Normalizer,
                    'features_to_use': self.features_to_use}

        with open(file_name, 'wb') as f:
            pickle.dump(classifier, f, pickle.HIGHEST_PROTOCOL)

    def load_classifier(self, file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        self.classifier = data['classifier']
        self.feature_options = data['feature_options']
        self.feature_Normalizer = data['feature_Normalizer']
        self.features_to_use = data['features_to_use']

    def color_hist(self, img):
        # Compute the histogram of the color channels separately
        nbins = self.__fop['hist_bins']
        bins_range = self.__fop['hist_range']
        channel1_hist = cv2.calcHist([img], [0], None, [nbins], list(bins_range))
        channel2_hist = cv2.calcHist([img], [1], None, [nbins], list(bins_range))
        channel3_hist = cv2.calcHist([img], [2], None, [nbins], list(bins_range))

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist, channel2_hist, channel3_hist)).squeeze()

        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def bin_spatial(self, img):
        size = self.__fop['spatial_size']
        color1 = cv2.resize(img[:,:,0], size).ravel()
        color2 = cv2.resize(img[:,:,1], size).ravel()
        color3 = cv2.resize(img[:,:,2], size).ravel()
        return np.hstack((color1, color2, color3))

    def extract_window_features(self, frame):
        #1) Define an empty list to receive features
        img_features = []

        #2) Convert the image colorspace
        img = getattr(frame,self.__fop['colorspace'])

        #3) compute spatial features, if requested, and add to feature list
        if self.__fop['spatial_feat']:
            spatial_features = self.bin_spatial(img)
            img_features.append(spatial_features)

        #4) Compute histogram features, if requested, and add to feature
        # list
        if self.__fop['hist_feat']:
            hist_features = self.color_hist(img)
            img_features.append(hist_features)

        #5) Compute hog features, if requested, and add to feature list
        hog_channel = self.__fop['hog_channel']
        if self.__fop['hist_feat']:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(img.shape[2]):
                    # hog_features.extent(self.get_hog_features(img[:,:,channel]))
                    hog_features.extent(self.hogDescriptor.compute(img[:,:,channel]).squeeze())
            else:
                # hog_features = self.get_hog_features(img[:,:,hog_channel])
                hog_features = self.hogDescriptor.compute(img[:,:,hog_channel]).squeeze()

            img_features.append(hog_features)

        #6) Return concatenated array of features
        return np.concatenate(img_features)


    def find_cars(self, frame, output_windows_searched=False): #, ystart=None, ystop=None, scale=1):

        if self.feature_Normalizer is None:
            print('Classifier has not yet been trained.')
            return None

        # Get image in correct colorspace
        img = getattr(frame,self.__fop['colorspace'])
        img_shape = img.shape[0:2]

        # Initialize parameters
        pix_per_cell = self.__fop['pix_per_cell']
        cell_per_block = self.__fop['cell_per_block']
        window = self.__fop['window']
        hog_channel = self.__fop['hog_channel']
        winSize = (window, window)

        X = []
        bboxes = []

        for srch_op in self.search_options:

            # Initialize parameters
            ystart = int(srch_op['roi'][0] * img_shape[0])
            ystop = int(srch_op['roi'][1] * img_shape[0])
            xstart = int(srch_op['roi'][2] * img_shape[1])
            xstop = int(srch_op['roi'][3] * img_shape[1])
            scale = srch_op['scale']
            cells_per_step = srch_op['cells_per_step']

            # Extract ROI
            search_img = img[ystart:ystop,xstart:xstop,:]

            # Scale
            if scale != 1:
                search_img = cv2.resize(search_img, (np.int(search_img.shape[1]/scale), np.int(search_img.shape[0]/scale)))

            # Compute window stride and the number of windows
            winStride = (cells_per_step*pix_per_cell, cells_per_step*pix_per_cell)
            num_windows = [(search_img.shape[0]-winSize[0])//winStride[0] + 1, (search_img.shape[1]-winSize[1])//winStride[1] + 1]

            # Extract HOG features
            if self.__fop['hist_feat']:
                if hog_channel == 'ALL':
                    hog = []
                    for channel in range(search_img.shape[2]):
                        hog_ch = self.hogDescriptor.compute(search_img[:,:,hog_channel],winStride).squeeze()
                        hog_ch = hog_ch.reshape(num_windows[0],num_windows[1],-1)
                        hog.append(hog_ch)
                    hog = np.concatenate(hog, axis=2)
                else:
                    hog = self.hogDescriptor.compute(search_img[:,:,hog_channel],winStride).squeeze()
                    hog = hog.reshape(max(1,num_windows[0]),max(1,num_windows[1]),-1)

            # Extract spatial color and color histogram features, and
            # combine with the HOG features
            for xb in range(num_windows[1]):
                for yb in range(num_windows[0]):

                    features = []

                    # Extract the image patch
                    xleft = xb * cells_per_step * pix_per_cell
                    ytop = yb * cells_per_step * pix_per_cell
                    subimg = search_img[ytop:ytop+window, xleft:xleft+window]

                    if window != 64:
                        subimg = cv2.resize(subimg, (64,64))

                    # Get color features
                    if self.__fop['spatial_feat']:
                        features.append(self.bin_spatial(subimg))

                    if self.__fop['hist_feat']:
                        features.append(self.color_hist(subimg))

                    # Extract HOG for this patch
                    if self.__fop['hist_feat']:
                        features.append(hog[yb,xb,:].squeeze())

                    X.append(np.concatenate(features))
                    xbox_left = int(xleft * scale)
                    ytop_draw = int(ytop * scale)
                    win_draw = int(window * scale)
                    bboxes.append(( xbox_left + xstart,
                                    ytop_draw + ystart,
                                    xbox_left + xstart + win_draw,
                                    ytop_draw + win_draw + ystart ))

        # Make prediction for all windows
        prediction = self.predict(np.array(X).astype(np.float64))
        bboxes = np.array(bboxes,dtype=int)

        # Keep bounding boxes predicted to be cars
        bboxes = bboxes[prediction==1,:]

        # Add bounding boxes to the looking glass and extract objects
        self.looking_glass.add_new_frame(bboxes, img_shape)
        objects = self.looking_glass.locate_objects(min_area=64*64)

        if output_windows_searched:
            return objects, len(X)
        else:
            return objects


def draw_boxes(img, bboxes, color=(0,0,255)):
    for box in range(bboxes.shape[0]):
        p1 = tuple((bboxes[box,0], bboxes[box,1]))
        p2 = tuple((bboxes[box,2], bboxes[box,3]))
        cv2.rectangle(img, p1, p2, (0,0,255), 6)

def draw_centroids(img, centroids):
    for box in range(centroids.shape[0]):
        p1 = tuple((int(centroids[box,0]), int(centroids[box,1])))
        cv2.circle(img, p1, 7, (255,0,0), -1)

def test_stuff():
    car_classifier = CarClassifier()
    car_classifier.load_classifier('./car_classifier.pickle')

    test_img = "./test_images/test3.jpg"
    # test_img = "K:/Udacity_CarND/vehicles/GTI_Right/image0031.png"
    frame = Frame(test_img)

    # features = car_classifier.extract_window_features(frame)
    # features = np.array(features).astype(np.float64).squeeze()
    # features = features[None,:]
    # score = car_classifier.predict(features)
    # print(score)

    tic()
    cars, num = car_classifier.find_cars(frame)
    print('Frame process time: ', toc())
    objects, glass = car_classifier.looking_glass.locate_objects(True)

    img = frame.RGB
    draw_boxes(img, cars['bboxes'])
    plt.imshow(glass)
    plt.colorbar()
    plt.show()

# if __name__ == '__main__':
    # test_stuff()
