from frame import Frame
from carClassifier import CarClassifier
from carClassifier import draw_boxes
from carClassifier import draw_centroids
from moviepy.editor import VideoFileClip
from copy import copy

from tic_toc import *

import cv2
import numpy as np

debug = False

car_classifier = CarClassifier()
car_classifier.load_classifier('./car_classifier.pickle')
car_classifier.looking_glass.history_size = 25

# Create regions to search for cars
soh = CarClassifier.search_options_helper
# Note the input is: cells_per_step, roi, scale
search_options = [soh(2, [0.6,0.8,0,1], 2),
                  soh(3, [0.5,0.7,0.05,0.95], 1)]#,
car_classifier.search_options = search_options

def addText(img, text, pos, size=2):
    img = cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, (0,0,0), 8)
    img = cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, (255,255,255), 2)
    return img

def get_new_roi(box,size):
    roi = [max(0, box[1]-32)/size[0], min(size[0], box[3]+64)/size[0], max(0,box[0]-32)/size[1], min(size[1], box[2]+64)/size[1]]
    return roi

def process_video(img):
    frame = Frame(img, colorspace='RGB')

    tic()
    cars, num = car_classifier.find_cars(frame, output_windows_searched=True)
    objects, glass = car_classifier.looking_glass.locate_objects(return_lookingGlass=True, threshold=3)

    fps = 1./toc()

    # Compute new search regions
    boxes = objects['bboxes']
    new_search_options = copy(search_options)
    for i in range(boxes.shape[0]):
        roi = get_new_roi(boxes[i,:], img.shape[0:2])
        scale = 2*max(0,(boxes[i,3]-500)/220) + 1
        scale = int(10*scale) / 10
        new_search_options.append(soh(2, roi, scale))

    car_classifier.search_options = new_search_options

    # Annotate image
    if debug:
        glass[glass>40]=40
        glass = (glass.astype(np.float)*255/40).astype(np.uint8)
        out_img = np.dstack((glass, glass, glass))

    else:
        out_img = frame.RGB

    draw_boxes(out_img, cars['bboxes'])
    # draw_centroids(out_img, cars['centroids'])
    out_img = addText(out_img, 'FPS: {:.0f}'.format(fps), (50,60), size=1)
    out_img = addText(out_img, 'Number of windows searched: {:.0f}'.format(num), (50,120), size=0.7)

    return out_img

# Process videos

video_in = './project_video.mp4'
video_out = './project_video_processed.mp4'

print('Processing {}'.format(video_in))

# Set debugging state and process the video
clip1 = VideoFileClip(video_in)
processed_clip = clip1.fl_image(process_video)
processed_clip.write_videofile(video_out, audio=False)
