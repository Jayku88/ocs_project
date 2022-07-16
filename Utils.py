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

def save_class_list(class_list, model_name, dataset_name):
    class_list.sort()
    target=open("./checkpoints/" + model_name + "_" + dataset_name + "_class_list.txt",'w')
    for c in class_list:
        target.write(c)
        target.write("\n")

def load_class_list(class_list_file):
    class_list = []
    with open(class_list_file, 'r') as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            class_list.append(row)
    class_list.sort()
    return class_list

def get_subfolders(directory):
    subfolders = os.listdir(directory)
    subfolders.sort()
    return subfolders

def get_num_files(directory):
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

def build_finetune_model(base_model, dropout, fc_layers, num_classes):

    x = base_model.output
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x)
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    return finetune_model

