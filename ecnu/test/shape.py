#!/usr/bin/env python
# coding=utf-8
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2, re
import argparse

class rectangle():


    def __init__(self, xmin, ymin, xmax, ymax):
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

    def getWidth(self):
        return self._xmax-self._xmin

    def getHeight(self):
        return self._ymax-self._ymin

    def getxmin(self):
        return self._xmin

    def getymin(self):
        return self._ymin

    def getxmax(self):
        return self._xmax

    def getymax(self):
        return self._ymax

    def getArea(self):
        return abs(self._xmax-self._xmin)*abs(self._ymax-self._ymin)

    def getCenter(self):

        centerx = self._xmin+self.getWidth()/2
        centery = self._ymin+self.getHeight()/2
        center=point(centerx, centery)
        return center


class point():
    def __init__(self, x, y):
        self._x=x
        self._y=y















































