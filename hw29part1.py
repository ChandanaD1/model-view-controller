# model view controller

import cv2 #image processing
import csv #csv files
import pandas as pd #create dataframe (table format)
import seaborn as sb #good graphs
import matplotlib.pyplot as plt #draw the graphs
from sklearn.datasets import fetch_openml #allows us to get data from openml library (has digit images)
from sklearn.model_selection import train_test_split #split data to train and test it
from sklearn.linear_model import LogisticRegression #logistic regression prediction model
from sklearn.metrics import accuracy_score #predict accuracy of our model
import numpy as np
from PIL import Image, ImageOps
import os,ssl,time

X =  np.load("image.npz")["arr_0"]
Y = pd.read_csv("hw27.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses = len(classes)

x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=9,train_size=3500,test_size=500)
model = LogisticRegression(solver="saga",multi_class="multinomial").fit(x_train,y_train)
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test,y_predict)
print(accuracy)

def get_prediction(image):
    image_pill = Image.open(image)
    img_l = image_pill.convert("L")
    img_resize = img_l.resize((28,28),Image.ANTIALIAS)
    img_invert = ImageOps.invert(img_resize)
    pixal_filter = 20
    min_pixal = np.percentile(img_invert,pixal_filter)
    img_clip = np.clip(img_invert-min_pixal,0,255)
    max_pixal = np.max(img_invert)
    img_scale = np.asarray(img_clip)/max_pixal
    test_sample = np.array(img_scale).reshape(1,784)
    test_predict = model.predict(test_sample)
    print(test_predict)
    return(test_predict[0])

