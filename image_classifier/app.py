#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:16:47 2018

@author: ayush
"""
#Usage: python app.py
import os
import tensorflow as tf
from flask import Flask, render_template, request                                                          
from werkzeug import secure_filename
from keras.preprocessing.image import  load_img, img_to_array
from keras.models import  load_model
import numpy as np

import time
import uuid
#import base64

img_width, img_height = 32,32
model_path = './model/model.h5'
#model_weights_path = './models/weights.h5'
model = load_model(model_path)
#model.load_weights(model_weights_path)
graph = tf.get_default_graph()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

#def get_as_base64(url):
 #   return base64.b64encode(request.get(url).content)

        
def predict(file):
    with graph.as_default():
        x = load_img(file, target_size=(img_width,img_height))
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        array = model.predict(x)
        result = array[0]
        answer = np.argmax(result)
        if answer == 0:
            print("Label: airplane")
        elif answer == 1:
        	    print("Label: automobile")
        elif answer == 2:
        	    print("Label: bird")
        elif answer == 3:
        	    print("Label: cat")
        elif answer == 4:
        	    print("Label: deer")
        elif answer == 5:
        	    print("Label: dog")
        elif answer == 6:
        	    print("Label: frog")
        elif answer == 7:
        	    print("Label: horse")
        elif answer == 8:
        	    print("Label: ship")
        elif answer == 9:
        	    print("Label: truck")
    return answer

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
       
        #start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            if result == 0:
                label = "airplane"
            elif result == 1:
                label = "automobile"
            elif result == 2:
                label =  "bird"
            elif result == 3:
            	    label=" cat"
            elif result == 4:
            	    label = "deer"
            elif result == 5:
            	    label= "dog"
            elif result == 6:
            	    label = "frog"
            elif result == 7:
            	    label =  "horse"
            elif result == 8:
            	    label = "ship"
            elif result == 9:
            	    label = "truck"
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=label, imagesource='../uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

#from werkzeug import SharedDataMiddleware
#app.add_url_rule('/uploads/<filename>', 'uploaded_file',
#                 build_only=True)
#app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
#    '/uploads':  app.config['UPLOAD_FOLDER']
#})

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)
