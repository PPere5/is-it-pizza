from importlib import import_module
import os
from flask import Flask, render_template, Response, url_for, escape, request, flash
import time
import pizza_functions
import cv2

from flask_wtf import FlaskForm
from wtforms import SubmitField, BooleanField


import tensorflow as tf
import numpy as np
import imageio
from PIL import Image
import keras.models
from keras.models import model_from_json

from multiprocessing import Value

import yolo




Camera = import_module('camera_opencv').Camera

button_counter = Value('i', 0)
app = Flask(__name__)


app.config['SECRET_KEY'] = 'fdc38d8dccf4a3c76631e137d99d386e'

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
	loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return loaded_model, graph

#predict if a frame is pizza

def predict_pizza(frame, button_counter_value):
    # import numpy as np
    # import cv2
    # import imageio
    # from PIL import Image

    if button_counter_value==0:
        global model, graph
        #initialize these variables
        model, graph = init()

    img_rows, img_cols = 200, 200

    print('Made it here')

    nparr = np.fromstring(frame, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1

    img_np = np.array(Image.fromarray(img_np).resize((img_rows, img_cols)))

    #cv2.imwrite('saved_frames/outfile'+str(button_counter_value)+'.png', img_np)

    # img_np = imresize(img_np,(img_rows, img_cols))
	#convert to a 4D tensor to feed into our model
    img_np = img_np.reshape(1,img_rows, img_cols,3)
	#in our computation graph
    with graph.as_default():
        #perform the prediction
        out = model.predict(img_np)
        print(out)

        out = np.argmax(out,axis=1)[0]

        if out==0:
            response=True
        else:
            response=False


    return response	





class PredictionForm(FlaskForm):
    submit = SubmitField('Predict')
 


@app.route('/', methods=['GET', 'POST'])
def index():
    """Video streaming home page."""
    form = PredictionForm()
    pred_data = form.submit.data
    mode='yolo'
    bool_pizza = False
    infos = pizza_functions.give_infos(bool_pizza)
    if pred_data:
        frame = Camera().get_frame()
        if mode=='keras':
            bool_pizza = predict_pizza(frame, button_counter.value)
        else:
            bool_pizza = yolo.pizza_yolo(frame, button_counter.value)
        infos = pizza_functions.give_infos(bool_pizza)
        form.submit.data = False
        pred_data = form.submit.data
        with button_counter.get_lock():
            button_counter.value += 1
    return render_template('index.html', infos=infos, form=form, pred_data = pred_data, count=button_counter.value)


@app.route('/info')
def about():
    return render_template('info.html', title='info')

def gen(camera):
    """Video streaming generator function."""
    global frame
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)






if __name__ == '__main__':
    app.run(threaded=True)

