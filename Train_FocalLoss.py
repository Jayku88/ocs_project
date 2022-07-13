from __future__ import print_function
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras import backend as K

from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import load_model

import numpy as np
import argparse
import random, glob
import os, sys, csv
import cv2
import time, datetime
import utils

def binary_focal_loss(gamma=2., alpha=.25):
   
    def binary_focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--dataset', type=str, default="oral", help='Dataset you are using.')
parser.add_argument('--resize_height', type=int, default=224, help='Height of cropped input image to network')
parser.add_argument('--resize_width', type=int, default=224, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
parser.add_argument('--dropout', type=float, default=1e-3, help='Dropout ratio')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--rotation', type=float, default=0.0, help='Whether to randomly rotate the image for data augmentation')
parser.add_argument('--zoom', type=float, default=0.0, help='Whether to randomly zoom in for data augmentation')
parser.add_argument('--shear', type=float, default=0.0, help='Whether to randomly shear in for data augmentation')
parser.add_argument('--model', type=str, default="MobileNet", help='Your pre-trained classification model of choice')
args = parser.parse_args()


BATCH_SIZE = args.batch_size
WIDTH = args.resize_width
HEIGHT = args.resize_height
FC_LAYERS = [1024, 1024]
TRAIN_DIR = args.dataset + "/train/"
VAL_DIR = args.dataset + "/val/"
TEST_DIR = args.dataset + "/val/"

preprocessing_function = None
base_model = None

from keras.applications.mobilenet import preprocess_input
preprocessing_function = preprocess_input
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

	
if args.mode == "train":
    print("Begin training...")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    print("Resize Height -->", args.resize_height)
    print("Resize Width -->", args.resize_width)
    print("Num Epochs -->", args.num_epochs)
    print("Batch Size -->", args.batch_size)

    print("Data Augmentation:")
    print("\tVertical Flip -->", args.v_flip)
    print("\tHorizontal Flip -->", args.h_flip)
    print("\tRotation -->", args.rotation)
    print("\tZooming -->", args.zoom)
    print("\tShear -->", args.shear)
    print("")

    if not os.path.isdir("checkpoints"):
        os.makedirs("checkpoints")

    train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocessing_function,
      rotation_range=args.rotation,
      shear_range=args.shear,
      zoom_range=args.zoom,
      horizontal_flip=args.h_flip,
      vertical_flip=args.v_flip
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)
    validation_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE,shuffle=False)
    test_generator = val_datagen.flow_from_directory(TEST_DIR, target_size=(HEIGHT, WIDTH), batch_size=1, shuffle=False)
	
    class_list = utils.get_subfolders(TRAIN_DIR)

    finetune_model = utils.build_finetune_model(base_model, dropout=args.dropout, fc_layers=FC_LAYERS, num_classes=len(class_list))

    for layer in finetune_model.layers[:]:
        layer.trainable=False
    #Freeze some layers

    adam = Adam(lr=)
    finetune_model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
	
    num_train_images = utils.get_num_files(TRAIN_DIR)
    num_val_images = utils.get_num_files(VAL_DIR)

    def lr_decay(epoch):
    #define learning rate schedule here

    learning_rate_schedule = LearningRateScheduler(lr_decay)

    filepath=""
    checkpoint = ModelCheckpoint(filepath=filepath, monitor="val_accuracy", verbose=2, mode='max', save_best_only=True)
    callbacks_list = [checkpoint]

    class_weight = {}
    history = finetune_model.fit_generator(train_generator, epochs=args.num_epochs, workers=1, steps_per_epoch=num_train_images // BATCH_SIZE, 
        validation_data=validation_generator, validation_steps=num_val_images // BATCH_SIZE, class_weight=class_weight, verbose=2, shuffle=False, use_multiprocessing=False, callbacks=[checkpoint])
    val_acc = history.history['val_accuracy']
    print("val_acc")
    print(val_acc)
    acc = history.history['accuracy']
    print("acc")
    print(acc)
    val_loss = history.history['val_loss']
    print("val_loss")
    print(val_loss)
    loss = history.history['loss']
    print("loss")
    print(loss)
	
