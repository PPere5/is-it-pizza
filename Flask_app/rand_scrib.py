from importlib import import_module
import os
from flask import Flask, render_template, Response, url_for, escape, request, flash
import time
import cv2

from flask_wtf import FlaskForm
from wtforms import SubmitField


import tensorflow as tf
import numpy as np
import imageio
from PIL import Image
import keras.models
from keras.models import model_from_json




Camera = import_module('camera_opencv').Camera


#load json model

def init(): 
	json_file = open('model.json','r')
	print("Loaded Model from disk 1/3")
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	print("Loaded Model from disk 2/3")
	#load woeights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded Model from disk 3/3")

	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return loaded_model, graph

img_rows, img_cols = 100, 100

image =np.array(cv2.resize(imageio.imread('outfile.jpg'), (img_rows, img_cols)))

global model, graph
#initialize these variables
model, graph = init()



img_np = image.reshape(1,img_rows, img_cols,3)
#in our computation graph
with graph.as_default():
	#perform the prediction
	out = model.predict(img_np)

out = np.argmax(out,axis=1)[0]

print(out)