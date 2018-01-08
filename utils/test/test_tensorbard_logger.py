import sys
import unittest
import numpy as np
import tensorflow as tf
import cv2

sys.path.append('../')
from utils.tensorboard_logger import Logger

class TestLogger(unittest.TestCase):

    def setUp(self):
        writer = tf.summary.FileWriter('/tmp/test/TestLogger')
        self.logger = Logger(writer)
        
    def test_add_images(self):
        img = cv2.imread('breakout_2_t7.20170.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = np.abs(img-255)
        self.assertFalse(np.all(img == img2))
        images = np.array([img, img2 , img, img])
        self.logger.add_images('test/image2', images)
        self.assertEqual(len(self.logger.values), 4)
        self.assertEqual(self.logger.values[0].tag, 'test/image2/image/0')
        self.assertEqual(self.logger.values[1].tag, 'test/image2/image/1')
        self.assertEqual(self.logger.values[2].tag, 'test/image2/image/2')
        self.assertEqual(self.logger.values[3].tag, 'test/image2/image/3')
 
        img = img[...,0:1]
        images = np.array([img, img , img, img])
        self.logger.add_images('test/image3', images)
        self.assertEqual(len(self.logger.values), 8)
        self.assertEqual(self.logger.values[4].tag, 'test/image3/image/0')
        self.assertEqual(self.logger.values[5].tag, 'test/image3/image/1')
        self.assertEqual(self.logger.values[6].tag, 'test/image3/image/2')
        self.assertEqual(self.logger.values[7].tag, 'test/image3/image/3')

        height, width, channel = img.shape
        img = img.reshape(height, width)
        images = np.array([img, img , img, img])
        self.logger.add_images('test/image4', images)
        self.assertEqual(len(self.logger.values), 12)
        self.assertEqual(self.logger.values[8].tag, 'test/image4/image/0')
        self.assertEqual(self.logger.values[9].tag, 'test/image4/image/1')
        self.assertEqual(self.logger.values[10].tag, 'test/image4/image/2')
        self.assertEqual(self.logger.values[11].tag, 'test/image4/image/3')
        self.logger.flush(4)
       