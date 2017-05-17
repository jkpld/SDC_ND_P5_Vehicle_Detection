import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from carClassifier import CarClassifier
from frame import Frame
from tic_toc import *

def extract_features(imgs, car_classifier):
    features = []
    for img in imgs:
        img_features = car_classifier.extract_window_features(Frame(img))
        features.append(img_features)

    return np.array(features).astype(np.float64)

def array_split(X, fraction=0.2):
    test_size = int(fraction*X.shape[0] - 0.5)
    split_pos = np.random.randint(0, X.shape[0]-test_size-1)

    X_test = X[split_pos:split_pos+test_size,:]
    X_train = np.delete(X, np.arange(split_pos, split_pos+test_size), 0)

    return X_train, X_test

def create_car_classifier(car_classifier, file_name, use_best_of_each_category=True, use_best=False):
    import glob

    test_size = 0.2

    # Train/test image files ==================================================
    gti_car_images = glob.glob('../vehicles/GTI_*/*.png')
    kit_car_images = glob.glob('../vehicles/KITTI_extracted/*.png')
    gti_notcar_images = glob.glob('../non-vehicles/GTI/*.png')
    extra_notcar_images = glob.glob('../non-vehicles/Extras/*.png')

    # Extract features ========================================================
    tic()
    gti_car_features = extract_features(gti_car_images, car_classifier)
    kit_car_features = extract_features(kit_car_images, car_classifier)
    gti_notcar_features = extract_features(gti_notcar_images, car_classifier)
    extra_notcar_features = extract_features(extra_notcar_images, car_classifier)
    print()
    print('Time to extract features: ', toc())

    # Create train and test data sets =========================================

    # Split gti features to use a random 20% chunk for testing
    gti_car_train, gti_car_test = array_split(gti_car_features, fraction=test_size)
    gti_notcar_train, gti_notcar_test = array_split(gti_notcar_features, fraction=test_size)

    # randomly split the kit and extra features
    kit_car_train, kit_car_test = train_test_split(kit_car_features, test_size=0.2)
    extra_notcar_train, extra_notcar_test = train_test_split(extra_notcar_features, test_size=0.2)

    # combine all the features
    X_train = np.vstack((gti_car_train, kit_car_train, gti_notcar_train, extra_notcar_train))
    X_test = np.vstack((gti_car_test, kit_car_test, gti_notcar_test, extra_notcar_test))
    y_train = np.hstack((np.ones(gti_car_train.shape[0]+kit_car_train.shape[0]), np.zeros(gti_notcar_train.shape[0]+extra_notcar_train.shape[0])))
    y_test = np.hstack((np.ones(gti_car_test.shape[0]+kit_car_test.shape[0]), np.zeros(gti_notcar_test.shape[0]+extra_notcar_test.shape[0])))

    # Shuffle the data
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    tic()
    car_classifier.train(X_train, y_train, use_best_of_each_category=use_best_of_each_category, use_best=use_best)
    print('Train time: ', toc())

    if car_classifier.features_to_use is None:
        print('Classifier using {} features'.format(X_train.shape[1]))
    else:
        print('Classifier using {}/{} features'.format(np.sum(car_classifier.features_to_use), X_train.shape[1]))

    tic()
    score = car_classifier.test(X_test,y_test)
    print('Classifier accuracy: {} (time={:.4f} sec for {} samples)'.format(score, toc(), X_test.shape[0]))
    print()
    car_classifier.save_classifier(file_name)


car_classifier = CarClassifier(classifier='AdaBoost')
create_car_classifier(car_classifier, './car_classifier.pickle', use_best_of_each_category=False, use_best=False)
