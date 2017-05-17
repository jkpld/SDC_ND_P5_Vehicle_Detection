## Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Extract features (spatial, color, HOG) and train a classifier for detecting cars.
* Implement a sliding window search to find cars in images.
* Use a history of previous car detections to help eliminate false positive car detections.
* Apply these methods to a video and draw bounding boxes around each detected car.

Here I will describe how I addressed each of the [goals](https://review.udacity.com/#!/rubrics/513/view) for this project.

---

### 1. Features

The feature extraction (and the majority of the code for this project) is contained in the class `CarClassifier` located in `carClassifier.py`. I used three types of features, all extracted from the Lab colorspace:

- _Spatial features_. Down-sampled 16x16 image of each window. `CarClassifier.bin_spatial()`
- _Color histogram features_. Histograms of the color from each color channel of the window with 32 bins per channel. `CarClassifier.color_hist()`
- _HOG features_. Histogram of oriented gradients with 9 orientations, 8 pixels per cell, and 2 cells per normalization block. HOG features were only extracted for the L color channel. `CarClassifier.hogDescriptor.compute()`

The total number of extracted features is 2628. The HOG features were computed using OpenCV's `HOGDescriptor` method, which I found to be much faster than the skimage method. Additionally, the color histograms were computed using OpenCV's `calcHist` function, which is also much faster than numpy's histogram method.

### 2. Classifier
I used the GTI and KITTI data sets of cars and non-cars with a 80/20 split for training/testing. For the GTI data sets, I extracted a random contiguous block of 20% for the testing to account for the fact that images near each other in the data set are approximately the same.

All features were normalized to zero norm and unit variance. I used an AdaBoost tree classifier that obtained an accuracy of 0.993 on the test set.

In order to make the classifier more robust to false positives, I moved the classifier threshold from 0.5 to 0.515. (A threshold above ~0.53 removes almost all car detections.)

The file containing the code for reading in the images, extracting features, and then training and saving the classifier is `create_car_classifier.py`.

### 3. Sliding window search
I perform a sliding window search over two regions:

- Search 1:
    - ROI: [0.6, 0.8, 0, 1]
    - Step size: 2 cells (16 pixels)
    - Search scale: 2 (128x128)
- Search 2:
    - ROI: [0.5, 0.7, 0.05, 0.95]
    - Step size: 3 cells (24 pixels)
    - Search scale: 1 (64x64)

where the ROI is [y-start, y-end, x-start, x-end] in fraction of image size. The first region detects larger cars and the second region detects smaller cars.

The purpose of these two regions (which produce a total of 221 search windows) is to find the general region of cars. Around each region of a car, I then add an additional search region that encompasses the general region. This allows for more robust detection of cars.

Note that the addition of search regions only occurs on subsequent image frames from a video, and therefore does not help single image frames. The code for adding additional search regions is in `process_video()` located in `process_video.py`.

The sliding window search is implemented in `CarClassifier.find_cars()`. Note that OpenCV's `HOGDescriptor` function automatically performs a sliding window search.

### 4. Video implementation
To make the car detection more robust, I used a used the history of previous car detections to generate better guess about where a car is (a heat map). This is implemented in the class `LookingGlass` located in `lookingGlass.py`.

I use a history size of 25 frames and say any region of the heat map with a value over 10 contains a car.

Any region of the heat map that has a value over 3, I assume may have a car and I create an additional search region around it for the next image frame. (See 3. Sliding window search)

The final results are shown in the processed videos,  [project video](./project_video_processed.mp4), and [project video debug (heat map)](./project_video_debug.mp4).

The first video shows the project video with cars annotated. I also show the FPS of the computation (around 9) and the number of windows searched. The second video shows the heat map with cars annotated for the same video.

### 5. Discussion
One big issue is "loosing" cars. For example, when the white car in the video gets far away, my method is not able to continue tracking it (even though it should be able too). This whole method could be made more robust with the addition of car tracking (using Kalman filters for example), where the expected position and size of a car can be estimated based on its history. This expected position can then be used to make better estimates about the car will be in the next frame.
