#!/usr/bin/env python3
"""This module contains the Yolo class
that uses the Yolo v3 algorithm to perform object detection
includes processing output method
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.activations import sigmoid  # type: ignore


class Yolo:
    """This class uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Yolo class constructor
        Args:
            model_path: path to a Darknet Keras model
            classes_path: path to list of class names
            class_t: box score threshold for initial filtering
            nms_t: IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs
        Args:
            outputs: list of numpy.ndarrays with predictions
            image_size: numpy.ndarray with original image size
        Returns: tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            box = output[..., :4]
            t_x = box[..., 0]
            t_y = box[..., 1]
            t_w = box[..., 2]
            t_h = box[..., 3]

            c_x = np.arange(grid_width).reshape(1, grid_width)
            c_x = np.repeat(c_x, grid_height, axis=0)
            c_x = np.repeat(c_x[..., np.newaxis], anchor_boxes, axis=2)

            c_y = np.arange(grid_width).reshape(1, grid_width)
            c_y = np.repeat(c_y, grid_height, axis=0).T
            c_y = np.repeat(c_y[..., np.newaxis], anchor_boxes, axis=2)

            b_x = (sigmoid(t_x) + c_x) / grid_width
            b_y = (sigmoid(t_y) + c_y) / grid_height

            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]

            image_width = self.model.input.shape[1]
            image_height = self.model.input.shape[2]
            b_w = (anchor_width * np.exp(t_w)) / image_width
            b_h = (anchor_height * np.exp(t_h)) / image_height

            x1, y1 = (b_x - b_w / 2), (b_y - b_h / 2)
            x2, y2 = (b_x + b_w / 2), (b_y + b_h / 2)

            x1 = x1 * image_size[1]
            y1 = y1 * image_size[0]
            x2 = x2 * image_size[1]
            y2 = y2 * image_size[0]

            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

            box_confidence = output[..., 4:5]
            box_confidence = 1 / (1 + np.exp(-box_confidence))
            box_confidences.append(box_confidence)

            box_class_prob = output[..., 5:]
            box_class_prob = 1 / (1 + np.exp(-box_class_prob))
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        # Step 1: Compute box scores by multiplying the box confidences with the class probabilities.
        box_scores = [conf * prob for conf, prob in zip(box_confidences, box_class_probs)]

        # Step 2: Create lists to hold the filtered boxes, their class, and their scores.
        filtered_boxes = []
        box_classes = []
        box_scores_list = []

        # Step 3: Iterate over each set of scores to filter out boxes.
        for score in box_scores:
            # Find the class with the highest score for each box.
            class_scores = np.max(score, axis=-1, keepdims=False)
            classes = np.argmax(score, axis=-1)

            # Filter out boxes with a score lower than a threshold (e.g., 0.5).
            filter_mask = class_scores >= 0.5

            # Apply the filter to the scores, classes, and boxes.
            filtered_boxes.append(boxes[np.nonzero(filter_mask)])
            box_classes.append(classes[np.nonzero(filter_mask)])
            box_scores_list.append(class_scores[np.nonzero(filter_mask)])

        # Step 4: Concatenate the lists to get the final arrays.
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores_list, axis=0)

        return filtered_boxes, box_classes, box_scores
