from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,make_response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from sklearn import svm
import soundfile as sf
# Define a flask app
app = Flask(__name__)


import pickle
gender = pickle.load(open('models/SavedKnnModelForGender', 'rb'))
sc = pickle.load(open('models/SavedScaller', 'rb'))
age = pickle.load(open('models/SavedStackingModelForAge', 'rb'))
labelencoder_y2 = pickle.load(open('models/SavedLableEncoderForAge', 'rb'))


## Model saved with Keras model.save()
#MODEL_PATH = 'models/SavedKnnModelForGender'

# Load your trained model
#model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('models/model_resnet.h5')
print('Model loaded. Check http://127.0.0.1:5000/') 


#def model_predict():
#    [Fs, xx] = audioBasicIO.read_audio_file("Voices/myVoice.mp3");
#    F, f_names = ShortTermFeatures.feature_extraction(xx, Fs, 0.050*Fs, 0.025*Fs);
#    zz=pd.DataFrame(F.tolist())
#    mean=zz.mean(axis=1)
#    mean=mean.to_frame()
#    mean=mean.transpose()
#    mean = mean.iloc[:, 0:34].values
#    y_pred = classifier.predict(mean)
#    return y_pred

    
    
#    img = image.load_img(img_path, target_size=(224, 224))
#
#    # Preprocessing the image
#    x = image.img_to_array(img)
#    # x = np.true_divide(x, 255)
#    x = np.expand_dims(x, axis=0)
#
#    # Be careful how your trained model deals with the input
#    # otherwise, it won't make correct prediction!
#    x = preprocess_input(x, mode='caffe')
#
#    preds = model.predict(x)
#    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
#    return render_template('index.html')
    return render_template('index.html')



#@app.route('/predict', methods=['GET', 'POST'])
#def upload():
#    if request.method == 'POST':
#        
#        # Get the file from post request
#        f = request.files['file']
#        file_path = os.path.join('Voices', secure_filename(f.filename))
#        
#        [Fs, xx] = audioBasicIO.read_audio_file(file_path);
#        F, f_names = ShortTermFeatures.feature_extraction(xx, Fs, 0.050*Fs, 0.025*Fs);
#        zz=pd.DataFrame(F.tolist())
#        mean=zz.mean(axis=1)
#        mean=mean.to_frame()
#        mean=mean.transpose()
#        mean = mean.iloc[:, 0:34].values     
#        # Feature Scaling
#        mean = sc.transform(mean)        
#        # Predicting the Test set results
#        genderPred = gender.predict(mean)
#        print(genderPred)  
#        
#        agePred = age.predict(mean)
#        agePred = labelencoder_y2.inverse_transform(agePred.astype('int'))
#        print(agePred)
#        a=[[genderPred],[agePred]]
#        p=str(a)
#        
#    
##        # Save the file to ./uploads
##        basepath = os.path.dirname(__file__)
##        file_path = os.path.join('Voices', secure_filename(f.filename))
##        f.save(file_path)
#
#        # Make prediction
##        preds = model_predict()
#
#        # Process your result for human
#        # pred_class = preds.argmax(axis=-1)            # Simple argmax
##        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
#        result = str(a)               # Convert to string
#        return result
#    return None

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        # Get the file from post request
        f = request.files['file']
        file_path = os.path.join('Voices', secure_filename(f.filename))
        f.save(file_path)
        
        [Fs, xx] = audioBasicIO.read_audio_file(file_path);
        F, f_names = ShortTermFeatures.feature_extraction(xx, Fs, 0.050*Fs, 0.025*Fs);
        zz=pd.DataFrame(F.tolist())
        mean=zz.mean(axis=1)
        mean=mean.to_frame()
        mean=mean.transpose()
        mean = mean.iloc[:, 0:34].values     
        # Feature Scaling
        mean = sc.transform(mean)        
        # Predicting the Test set results
        genderPred = gender.predict(mean)
        genderPred= str(genderPred)  
        
        agePred = age.predict(mean)
        agePred = labelencoder_y2.inverse_transform(agePred.astype('int'))
        agePred=str(agePred)
        a={1:genderPred,2:agePred}
        
        
    
#        # Save the file to ./uploads
#        basepath = os.path.dirname(__file__)
#        file_path = os.path.join('Voices', secure_filename(f.filename))
        

        # Make prediction
#        preds = model_predict()

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
#        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
                      # Convert to string
        
        return a
    return None


if __name__ == '__main__':
    app.run(debug=True)

