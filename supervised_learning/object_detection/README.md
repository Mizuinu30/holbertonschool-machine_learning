
## Object Detection

## General

# Object Detection and Image Processing Concepts

## What is OpenCV and how do you use it?
OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It contains more than 2500 optimized algorithms, which can be used for a variety of tasks including detecting and recognizing faces, identifying objects, classifying human actions in videos, tracking camera movements, and extracting 3D models of objects. OpenCV is widely used in both academia and industry for its comprehensive functionality and performance.

### Usage:
To use OpenCV, you need to install the library (`pip install opencv-python`) and import it into your Python script. Basic operations include reading images, converting color spaces, applying filters, and detecting edges.

```python
import cv2

# Load an image
image = cv2.imread('image.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the image
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## What is object detection?
Object detection is a computer vision technique used to identify and locate objects within an image or video. This involves not only classifying objects but also drawing bounding boxes around them to indicate their positions. Object detection algorithms are crucial in various applications, such as surveillance, autonomous driving, and image retrieval.

## What is the Sliding Windows algorithm?
The Sliding Windows algorithm is a technique used in object detection where a fixed-size window slides over the input image to detect objects. At each position of the window, a classifier determines whether the window contains the object of interest. This method ensures that objects of different scales and aspect ratios can be detected by resizing the window or the input image.

## What is a single-shot detector?
A single-shot detector (SSD) is a type of object detection algorithm that detects objects in images in a single forward pass of the network. Unlike traditional methods that use region proposals and multiple stages, SSD combines predictions of different aspect ratios and scales from multiple feature maps to achieve high accuracy and real-time performance.

## What is the YOLO algorithm?
YOLO (You Only Look Once) is a state-of-the-art, real-time object detection algorithm. YOLO divides the input image into a grid and predicts bounding boxes and probabilities for each grid cell. It treats object detection as a single regression problem, directly predicting class probabilities and bounding box coordinates. YOLO is known for its speed and accuracy.

## What is IOU and how do you calculate it?
Intersection over Union (IOU) is a metric used to evaluate the accuracy of an object detector on a particular dataset. IOU calculates the overlap between two bounding boxes: the predicted bounding box and the ground truth bounding box. It is defined as the area of the intersection divided by the area of the union of the two bounding boxes.

### Calculation:
Given two bounding boxes A and B:
```python
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou
```

## What is non-max suppression?
Non-max suppression is a technique used in object detection to filter out redundant bounding boxes. When multiple bounding boxes are predicted for the same object, non-max suppression keeps only the box with the highest confidence score and eliminates the others based on their IOU values.

### Steps:
1. Select the box with the highest confidence score.
2. Compare this box with the remaining boxes and remove those with IOU greater than a threshold.
3. Repeat the process for the next highest confidence box.

## What are anchor boxes?
Anchor boxes are predefined bounding boxes of different sizes and aspect ratios used in object detection models like SSD and YOLO. They serve as references to predict the location and size of objects in the image. The model learns to adjust these anchor boxes to better fit the ground truth boxes during training.

## What is mAP and how do you calculate it?
Mean Average Precision (mAP) is a metric used to evaluate the performance of object detection algorithms. It averages the precision of each class at different recall levels. Precision and recall are calculated from the true positives, false positives, and false negatives. The mAP is the mean of the average precision values for all classes.

### Calculation:
1. Compute precision and recall for each class.
2. Plot the precision-recall curve.
3. Calculate the area under the curve (AUC) for each class.
4. Compute the mean of the AUCs across all classes.

```python
from sklearn.metrics import precision_recall_curve, auc

# Example precision and recall arrays
precision = [0.9, 0.8, 0.7, 0.6]
recall = [0.1, 0.2, 0.3, 0.4]

# Calculate AUC for the precision-recall curve
auc_score = auc(recall, precision)
```
# Tasks

### Description
0. Initialize YolomandatoryWrite a classYolothat uses the Yolo v3 algorithm to perform object detection:class constructor:def __init__(self, model_path, classes_path, class_t, nms_t, anchors):model_pathis the path to where a Darknet Keras model is storedclasses_pathis the path to where the list of class names used for the Darknet model, listed in order of index, can be foundclass_tis a float representing the box score threshold for the initial filtering stepnms_tis a float representing the IOU threshold for non-max suppressionanchorsis anumpy.ndarrayof shape(outputs, anchor_boxes, 2)containing all of the anchor boxes:outputsis the number of outputs (predictions) made by the Darknet modelanchor_boxesis the number of anchor boxes used for each prediction2=>[anchor_box_width, anchor_box_height]Public instance attributes:model: the Darknet Keras modelclass_names: a list of the class names for the modelclass_t: the box score threshold for the initial filtering stepnms_t: the IOU threshold for non-max suppressionanchors: the anchor boxesroot@alexa-ml2:~/object_detection# cat 0-main.py
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('0-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
    yolo.model.summary()
    print('Class names:', yolo.class_names)
    print('Class threshold:', yolo.class_t)
    print('NMS threshold:', yolo.nms_t)
    print('Anchor boxes:', yolo.anchors)
root@alexa-ml2:~/object_detection# ./0-main.py
WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to
==================================================================================================
 input_1 (InputLayer)        [(None, 416, 416, 3)]        0         []

 conv2d (Conv2D)             (None, 416, 416, 32)         864       ['input_1[0][0]']

 batch_normalization (Batch  (None, 416, 416, 32)         128       ['conv2d[0][0]']
 Normalization)

 leaky_re_lu (LeakyReLU)     (None, 416, 416, 32)         0         ['batch_normalization[0][0]']

 zero_padding2d (ZeroPaddin  (None, 417, 417, 32)         0         ['leaky_re_lu[0][0]']
 g2D)

 conv2d_1 (Conv2D)           (None, 208, 208, 64)         18432     ['zero_padding2d[0][0]']

 batch_normalization_1 (Bat  (None, 208, 208, 64)         256       ['conv2d_1[0][0]']
 chNormalization)

 leaky_re_lu_1 (LeakyReLU)   (None, 208, 208, 64)         0         ['batch_normalization_1[0][0]'
                                                                    ]

 conv2d_2 (Conv2D)           (None, 208, 208, 32)         2048      ['leaky_re_lu_1[0][0]']

 batch_normalization_2 (Bat  (None, 208, 208, 32)         128       ['conv2d_2[0][0]']
 chNormalization)

 leaky_re_lu_2 (LeakyReLU)   (None, 208, 208, 32)         0         ['batch_normalization_2[0][0]'

..............

 leaky_re_lu_57 (LeakyReLU)  (None, 13, 13, 1024)         0         ['batch_normalization_57[0][0]
                                                                    ']

 leaky_re_lu_64 (LeakyReLU)  (None, 26, 26, 512)          0         ['batch_normalization_64[0][0]
                                                                    ']

 leaky_re_lu_71 (LeakyReLU)  (None, 52, 52, 256)          0         ['batch_normalization_71[0][0]
                                                                    ']

 conv2d_58 (Conv2D)          (None, 13, 13, 255)          261375    ['leaky_re_lu_57[0][0]']

 conv2d_66 (Conv2D)          (None, 26, 26, 255)          130815    ['leaky_re_lu_64[0][0]']

 conv2d_74 (Conv2D)          (None, 52, 52, 255)          65535     ['leaky_re_lu_71[0][0]']

 reshape (Reshape)           (None, 13, 13, 3, 85)        0         ['conv2d_58[0][0]']

 reshape_1 (Reshape)         (None, 26, 26, 3, 85)        0         ['conv2d_66[0][0]']

 reshape_2 (Reshape)         (None, 52, 52, 3, 85)        0         ['conv2d_74[0][0]']

==================================================================================================
Total params: 62001757 (236.52 MB)
Trainable params: 61949149 (236.32 MB)
Non-trainable params: 52608 (205.50 KB)
__________________________________________________________________________________________________
Class names: ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
Class threshold: 0.6
NMS threshold: 0.5
Anchor boxes: [[[116  90]
  [156 198]
  [373 326]]

 [[ 30  61]
  [ 62  45]
  [ 59 119]]

 [[ 10  13]
  [ 16  30]
  [ 33  23]]]
root@alexa-ml2:~/object_detection#Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/object_detectionFile:0-yolo.pyHelp×Students who are done with "0. Initialize Yolo"Review your work×Correction of "0. Initialize Yolo"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

1. Process OutputsmandatoryWrite a classYolo(Based on0-yolo.py):Add the public methoddef process_outputs(self, outputs, image_size):outputsis a list ofnumpy.ndarrays containing the predictions from the Darknet model for a single image:Each output will have the shape(grid_height, grid_width, anchor_boxes, 4 + 1 + classes)grid_height&grid_width=> the height and width of the grid used for the outputanchor_boxes=> the number of anchor boxes used4=>(t_x, t_y, t_w, t_h)1=>box_confidenceclasses=> class probabilities for all classesimage_sizeis anumpy.ndarraycontaining the image’s original size[image_height, image_width]Returns a tuple of(boxes, box_confidences, box_class_probs):boxes: a list ofnumpy.ndarrays of shape(grid_height, grid_width, anchor_boxes, 4)containing the processed boundary boxes for each output, respectively:4=>(x1, y1, x2, y2)(x1, y1, x2, y2)should represent the boundary box relative to original imagebox_confidences: a list ofnumpy.ndarrays of shape(grid_height, grid_width, anchor_boxes, 1)containing the box confidences for each output, respectivelybox_class_probs: a list ofnumpy.ndarrays of shape(grid_height, grid_width, anchor_boxes, classes)containing the box’s class probabilities for each output, respectivelyHINT1: The Darknet model is an input to the class for a reason. It may not always have the same number of outputs, input sizes, etc.HINT2: An explanatory video that might help :LINK.root@alexa-ml2:~/object_detection# cat 1-main.py
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('1-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
    output1 = np.random.randn(13, 13, 3, 85)
    output2 = np.random.randn(26, 26, 3, 85)
    output3 = np.random.randn(52, 52, 3, 85)
    boxes, box_confidences, box_class_probs = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))
    print('Boxes:', boxes)
    print('Box confidences:', box_confidences)
    print('Box class probabilities:', box_class_probs)
root@alexa-ml2:~/object_detection# ./1-main.py
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
Boxes: [array([[[[-2.13743365e+02, -4.85478868e+02,  3.05682061e+02,
           5.31534670e+02],
         [-6.28222336e+01, -1.13713822e+01,  1.56452678e+02,
           7.01966357e+01],
         [-7.00753664e+02, -7.99011810e+01,  7.77777040e+02,
           1.24440730e+02]],

                ...

        [[ 5.61142819e+02,  3.07685110e+02,  7.48124593e+02,
           6.41890468e+02],
         [ 5.80033260e+02,  2.88627445e+02,  7.62480440e+02,
           6.47683922e+02],
         [ 3.58437752e+02,  2.46899004e+02,  9.55713550e+02,
           7.08803200e+02]]]]), array([[[[-1.23432804e+01, -3.22999818e+02,  2.20307323e+01,
           3.46429534e+02],
         [-1.80604912e+01, -7.11557928e+00,  3.29117181e+01,
           3.76250584e+01],
         [-1.60588925e+02, -1.42683911e+02,  1.73612755e+02,
           1.73506691e+02]],

                ...

        [[ 6.68829175e+02,  4.87753124e+02,  6.82031480e+02,
           5.03135980e+02],
         [ 4.34438370e+02,  4.78337823e+02,  9.46316108e+02,
           5.09664332e+02],
         [ 6.22075514e+02,  4.82103466e+02,  7.35368626e+02,
           4.95580409e+02]]]]), array([[[[-2.69962831e+00, -7.76450289e-01,  1.24050569e+01,
           2.78936573e+00],
         [-4.22907727e+00, -1.06544368e+01,  1.17677684e+01,
           2.57860099e+01],
         [-7.89233952e+01, -2.81791298e+00,  8.90320793e+01,
           1.29705044e+01]],

                ...

        [[ 6.76849818e+02,  4.46289490e+02,  7.00081017e+02,
           5.41566130e+02],
         [ 6.66112867e+02,  4.81296831e+02,  7.28887543e+02,
           5.01093787e+02],
         [ 6.29818872e+02,  4.58019128e+02,  7.50570597e+02,
           5.32394185e+02]]]])]
Box confidences: [array([[[[0.86617546],
         [0.74162884],
         [0.26226237]],

                ...

        [[0.75932849],
         [0.53997516],
         [0.54609635]]]]), array([[[[0.56739016],
         [0.88809651],
         [0.58774731]],

                ...

        [[0.58788139],
         [0.3982885 ],
         [0.56051201]]]]), array([[[[0.73336558],
         [0.57887604],
         [0.59277021]],

                ...

        [[0.19774838],
         [0.69947049],
         [0.36158221]]]])]
Box class probabilities: [array([[[[0.27343225, 0.72113296, 0.46223277, ..., 0.6143566 ,
          0.177082  , 0.81581579],
         [0.40054928, 0.77249355, 0.55188134, ..., 0.17584277,
          0.76638851, 0.57857896],
         [0.66409448, 0.30929663, 0.33413323, ..., 0.53542882,
          0.42083943, 0.66630914]],

                ...

        [[0.48321664, 0.22789979, 0.6691948 , ..., 0.53945861,
          0.30098012, 0.72069712],
         [0.45435778, 0.68714784, 0.60871616, ..., 0.51156461,
          0.14100029, 0.38924579],
         [0.14533179, 0.91481357, 0.46558833, ..., 0.29158615,
          0.26098354, 0.6078719 ]]]]), array([[[[0.16748018, 0.74425493, 0.77271059, ..., 0.82238314,
          0.45290514, 0.35835817],
         [0.35157962, 0.88467242, 0.18324688, ..., 0.63359015,
          0.40203054, 0.48139226],
         [0.45924026, 0.81674653, 0.68472278, ..., 0.45661086,
          0.13878263, 0.58812507]],

                ...

        [[0.34600385, 0.2430844 , 0.82184407, ..., 0.64286074,
          0.2806551 , 0.10224861],
         [0.89692404, 0.22950708, 0.46779974, ..., 0.6787069 ,
          0.25042145, 0.63684789],
         [0.36853257, 0.67040213, 0.4840967 , ..., 0.4530742 ,
          0.20817072, 0.59335632]]]]), array([[[[0.5651767 , 0.5976206 , 0.38176215, ..., 0.27546563,
          0.31863509, 0.31522224],
         [0.34311079, 0.35702272, 0.52498233, ..., 0.67677487,
          0.13292343, 0.79922556],
         [0.31009487, 0.12745849, 0.45797284, ..., 0.20871353,
          0.60219055, 0.29796099]],

                ...

        [[0.24181667, 0.17739203, 0.77300902, ..., 0.59600114,
          0.3446732 , 0.75570862],
         [0.53289017, 0.31652626, 0.11990411, ..., 0.45424862,
          0.2372848 , 0.26196449],
         [0.40272803, 0.24201719, 0.80157031, ..., 0.2338579 ,
          0.18169015, 0.36450041]]]])]
root@alexa-ml2:~/object_detection#Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/object_detectionFile:1-yolo.pyHelp×Students who are done with "1. Process Outputs"Review your work×Correction of "1. Process Outputs"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

2. Filter BoxesmandatoryWrite a classYolo(Based on1-yolo.py):Add the public methoddef filter_boxes(self, boxes, box_confidences, box_class_probs):boxes: a list ofnumpy.ndarrays of shape(grid_height, grid_width, anchor_boxes, 4)containing the processed boundary boxes for each output, respectivelybox_confidences: a list ofnumpy.ndarrays of shape(grid_height, grid_width, anchor_boxes, 1)containing the processed box confidences for each output, respectivelybox_class_probs: a list ofnumpy.ndarrays of shape(grid_height, grid_width, anchor_boxes, classes)containing the processed box class probabilities for each output, respectivelyReturns a tuple of(filtered_boxes, box_classes, box_scores):filtered_boxes: anumpy.ndarrayof shape(?, 4)containing all of the filtered bounding boxes:box_classes: anumpy.ndarrayof shape(?,)containing the class number that each box infiltered_boxespredicts, respectivelybox_scores: anumpy.ndarrayof shape(?)containing the box scores for each box infiltered_boxes, respectivelyroot@alexa-ml2:~/object_detection# cat 2-main.py
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('2-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
    output1 = np.random.randn(13, 13, 3, 85)
    output2 = np.random.randn(26, 26, 3, 85)
    output3 = np.random.randn(52, 52, 3, 85)
    boxes, box_confidences, box_class_probs = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))
    boxes, box_classes, box_scores = yolo.filter_boxes(boxes, box_confidences, box_class_probs)
    print('Boxes:', boxes)
    print('Box classes:', box_classes)
    print('Box scores:', box_scores)
root@alexa-ml2:~/object_detection# ./2-main.py
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
Boxes: [[-213.74336488 -485.47886784  305.68206077  531.53467019]
 [ -62.82223363  -11.37138215  156.45267787   70.19663572]
 [ 190.62733946    7.65943712  319.201764     43.75737906]
 ...
 [ 647.78041714  491.58472667  662.00736941  502.60750466]
 [ 586.27543101  487.95333873  715.85860922  499.39422783]
 [ 666.1128673   481.29683099  728.88754319  501.09378706]]
Box classes: [19 54 29 ... 63 25 46]
Box scores: [0.7850503  0.67898563 0.81301861 ... 0.8012832  0.61427808 0.64562072]
root@alexa-ml2:~/object_detection#Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/object_detectionFile:2-yolo.pyHelp×Students who are done with "2. Filter Boxes"Review your work×Correction of "2. Filter Boxes"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

3. Non-max SuppressionmandatoryWrite a classYolo(Based on2-yolo.py):Add the public methoddef non_max_suppression(self, filtered_boxes, box_classes, box_scores):filtered_boxes: anumpy.ndarrayof shape(?, 4)containing all of the filtered bounding boxes:box_classes: anumpy.ndarrayof shape(?,)containing the class number for the class thatfiltered_boxespredicts, respectivelybox_scores: anumpy.ndarrayof shape(?)containing the box scores for each box infiltered_boxes, respectivelyReturns a tuple of(box_predictions, predicted_box_classes, predicted_box_scores):box_predictions: anumpy.ndarrayof shape(?, 4)containing all of the predicted bounding boxes ordered by class and box scorepredicted_box_classes: anumpy.ndarrayof shape(?,)containing the class number forbox_predictionsordered by class and box score, respectivelypredicted_box_scores: anumpy.ndarrayof shape(?)containing the box scores forbox_predictionsordered by class and box score, respectivelyroot@alexa-ml2:~/object_detection# cat 3-main.py
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('3-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
    output1 = np.random.randn(13, 13, 3, 85)
    output2 = np.random.randn(26, 26, 3, 85)
    output3 = np.random.randn(52, 52, 3, 85)
    boxes, box_confidences, box_class_probs = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))
    boxes, box_classes, box_scores = yolo.filter_boxes(boxes, box_confidences, box_class_probs)
    boxes, box_classes, box_scores = yolo.non_max_suppression(boxes, box_classes, box_scores)
    print('Boxes:', boxes)
    print('Box classes:', box_classes)
    print('Box scores:', box_scores)
root@alexa-ml2:~/object_detection# ./3-main.py
WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
Boxes: [[483.49145347 128.010205   552.78146847 147.87465464]
 [-38.91328475 332.66704009 102.94594841 363.78584864]
 [ 64.10861893 329.13266621 111.87941603 358.37523958]
 ...
 [130.0729606  467.20024928 172.42160784 515.90336094]
 [578.82381106  76.25699693 679.22893305 104.63320075]
 [169.12132771 304.32765204 251.1457077  342.16397829]]
Box classes: [ 0  0  0 ... 79 79 79]
Box scores: [0.80673525 0.80405611 0.78972362 ... 0.61758194 0.61455015 0.6001824 ]
root@alexa-ml2:~/object_detection#Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/object_detectionFile:3-yolo.pyHelp×Students who are done with "3. Non-max Suppression"Review your work×Correction of "3. Non-max Suppression"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

4. Load imagesmandatoryWrite a classYolo(Based on3-yolo.py):Add the static methoddef load_images(folder_path):folder_path: a string representing the path to the folder holding all the images to loadReturns a tuple of(images, image_paths):images: a list of images asnumpy.ndarraysimage_paths: a list of paths to the individual images inimagesroot@alexa-ml2:~/object_detection# cat 4-main.py
#!/usr/bin/env python3

if __name__ == '__main__':
    import cv2
    import numpy as np
    Yolo = __import__('4-yolo').Yolo

    np.random.seed(2)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
    images, image_paths = yolo.load_images('yolo_images/yolo/')
    image_paths, images = zip(*sorted(zip(image_paths, images)))
    i = np.random.randint(0, len(images))
    print(i)
    print(image_paths[i])
    cv2.imshow(image_paths[i], images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
root@alexa-ml2:~/object_detection# ./4-main.py
WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
0
yolo_images/yolo/dog.jpgRepo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/object_detectionFile:4-yolo.pyHelp×Students who are done with "4. Load images"Review your work×Correction of "4. Load images"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

5. Preprocess imagesmandatoryWrite a classYolo(Based on4-yolo.py):Add the public methoddef preprocess_images(self, images):images: a list of images asnumpy.ndarraysResize the images with inter-cubic interpolationRescale all images to have pixel values in the range[0, 1]Returns a tuple of(pimages, image_shapes):pimages: anumpy.ndarrayof shape(ni, input_h, input_w, 3)containing all of the preprocessed imagesni: the number of images that were preprocessedinput_h: the input height for the Darknet modelNote: this can vary by modelinput_w: the input width for the Darknet modelNote: this can vary by model3: number of color channelsimage_shapes: anumpy.ndarrayof shape(ni, 2)containing the original height and width of the images2=>(image_height, image_width)root@alexa-ml2:~/object_detection# cat 5-main.py
#!/usr/bin/env python3

if __name__ == '__main__':
    import cv2
    import numpy as np
    Yolo = __import__('5-yolo').Yolo

    np.random.seed(2)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
    images, image_paths = yolo.load_images('yolo_images/yolo/')
    image_paths, images = zip(*sorted(zip(image_paths, images)))
    pimages, image_shapes = yolo.preprocess_images(images)
    print(type(pimages), pimages.shape)
    print(type(image_shapes), image_shapes.shape)
    i = np.random.randint(0, len(images))
    print(images[i].shape, ':', image_shapes[i])
    cv2.imshow(image_paths[i], pimages[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
root@alexa-ml2:~/object_detection# ./5-main.py
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.
<class 'numpy.ndarray'> (6, 416, 416, 3)
<class 'numpy.ndarray'> (6, 2)
(576, 768, 3) : [576 768]Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/object_detectionFile:5-yolo.pyHelp×Students who are done with "5. Preprocess images"Review your work×Correction of "5. Preprocess images"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

6. Show boxesmandatoryWrite a classYolo(Based on5-yolo.py):Add the public methoddef show_boxes(self, image, boxes, box_classes, box_scores, file_name):image: anumpy.ndarraycontaining an unprocessed imageboxes: anumpy.ndarraycontaining the boundary boxes for the imagebox_classes: anumpy.ndarraycontaining the class indices for each boxbox_scores: anumpy.ndarraycontaining the box scores for each boxfile_name: the file path where the original image is storedDisplays the image with all boundary boxes, class names, and box scores(see example below)Boxes should be drawn as with a blue line of thickness 2Class names and box scores should be drawn above each box in redBox scores should be rounded to 2 decimal placesText should be written 5 pixels above the top left corner of the boxText should be written inFONT_HERSHEY_SIMPLEXFont scale should be 0.5Line thickness should be 1You should useLINE_AAas the line typeThe window name should be the same asfile_nameIf theskey is pressed:The image should be saved in the directorydetections, located in the current directoryIfdetectionsdoes not exist, create itThe saved image should have the file namefile_nameThe image window should be closedIf any key besidessis pressed, the image window should be closed without savingroot@alexa-ml2:~/object_detection# cat 6-main.py
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('6-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
    images, image_paths = yolo.load_images('yolo_images/yolo/')
    boxes = np.array([[119.22100287, 118.62197718, 567.75985556, 440.44121152],
                      [468.53530752, 84.48338278, 696.04923556, 167.98947829],
                      [124.2043716, 220.43365057, 319.4254314 , 542.13706101]])
    box_scores = np.array([0.99537075, 0.91536146, 0.9988506])
    box_classes = np.array([1, 7, 16])
    ind = 0
    for i, name in enumerate(image_paths):
        if "dog.jpg" in name:
            ind = i
            break
    yolo.show_boxes(images[i], boxes, box_classes, box_scores, "dog.jpg")
root@alexa-ml2:~/object_detection# ./6-main.py
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.press thesbuttonroot@alexa-ml2:~/object_detection# ls detections
dog.jpg
root@alexa-ml2-1:~/object_detection#Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/object_detectionFile:6-yolo.pyHelp×Students who are done with "6. Show boxes"0/9pts

7. PredictmandatoryWrite a classYolo(Based on6-yolo.py):Add the public methoddef predict(self, folder_path):folder_path: a string representing the path to the folder holding all the images to predictAll image windows should be named after the corresponding image filename without its full path(see examples below)Displays all images using theshow_boxesmethodReturns: a tuple of(predictions, image_paths):predictions: a list of tuples for each image of(boxes, box_classes, box_scores)image_paths: a list of image paths corresponding to each prediction inpredictionsroot@alexa-ml2:~/object_detection# cat 7-main.py
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('7-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
    predictions, image_paths = yolo.predict('yolo_images/yolo/')
    for i, name in enumerate(image_paths):
        if "dog.jpg" in name:
            ind = i
            break
    print(image_paths[ind])
    print(predictions[ind])
root@alexa-ml2:~/object_detection# ./7-main.py
WARNING:tensorflow:No training configuration found in save file: the model was *not* compiled. Compile it manually.pressspressspressspressspressspresssyolo_images/yolo/dog.jpg
(array([[119.10217975, 118.63844066, 567.89385531, 440.58719252],
       [468.68077557,  84.48196691, 695.97415748, 168.00746345],
       [124.10609986, 220.43732858, 319.45648934, 542.3966693 ]]), array([ 1,  7, 16]), array([0.99545544, 0.91439807, 0.99883264]))
root@alexa-ml2:~/object_detection#  ls detections
dog.jpg  eagle.jpg  giraffe.jpg  horses.jpg  person.jpg  takagaki.jpg
root@alexa-ml2-1:~/object_detection#Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/object_detectionFile:7-yolo.pyHelp×Students who are done with "7. Predict"0/7pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Object_Detection.md`