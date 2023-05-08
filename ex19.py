# Date: 07/06/2018

# Demo - Faster R-CNN with ResNet-101

# git clone https://github.com/tensorflow/models.git

import numpy as np
import os
import sys
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import random
import time
from utils import label_map_util

# in order to load a pre-trained model for prediction:
    #load graph
    def load_and_create_graph(path_to_pb):
        """
        Loads pre-trained grapgh from .pb file.
        path_to_pb: path to saved .pb file
        Tensorflow keeps graph global so nothing is returned"""

        with tf.gfile.FastGFile(path_to_pb, 'rb') as f:
            # initialize graph definition
            graph_def = tf.GraphDef()