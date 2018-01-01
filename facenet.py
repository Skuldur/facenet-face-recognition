from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import win32com.client as wincl

speak = wincl.Dispatch("SAPI.SpVoice")

PADDING = 50
ready = True

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

print("Total Params:", FRmodel.count_params())

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

def prepare_database():
    database = {}
    database["Sigurdur"] = img_to_encoding("images/sigurdur2.jpg", FRmodel)
    database["Kristin"] = img_to_encoding("images/kristin.jpg", FRmodel)
    database["Skuli"] = img_to_encoding("images/skuli3.jpg", FRmodel)
    return database

def webcam_face_detector(database):
    global ready

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    while rval:
        rval, frame = vc.read()
        img = frame

        if(ready):
            img = process_frame(img, frame, face_cascade)   
        
        key = cv2.waitKey(100)
        cv2.imshow("preview", img)

        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")

def process_frame(img, frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame,(x-PADDING,y-PADDING),(x+w+PADDING,y+h+PADDING),(255,0,0),2)
        find_identity(frame, x, y, w, h)

    return img

def find_identity(frame, x, y, w, h):
    global ready
    pool = Pool(processes=1) 
    height, width, channels = frame.shape
    part_image = frame[max(0, y-PADDING):min(height, y+h+PADDING), max(0, x-PADDING):min(width, x+w+PADDING)]
    
    identity = who_is_it(part_image, database, FRmodel)

    if identity is not None:
        ready = False
        pool.apply_async(welcome_guest, [identity])

def who_is_it(image, database, model):
    encoding = process_image(image, model)
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(db_enc - encoding)

        print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.55:
        return None
    else:
        return str(identity)

def welcome_guest(identity):
    global ready
    welcome_message = "Welcome %s. Have a nice day!" % identity
    speak.Speak(welcome_message)
    ready = True

if __name__ == "__main__":
    database = prepare_database()
    webcam_face_detector(database)

# ### References:
# 
# - Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
# - Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
# - The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
# - Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
# 
