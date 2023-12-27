
from flask import Flask, request, render_template, jsonify
import base64


import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('improved_model.h5')
model.make_predict_function()


app = Flask(__name__)


import cv2
import numpy as np

import matplotlib.pyplot as plt


@app.route('/')
def index():
    return render_template('/index.html')



@app.route('/recognize', methods=['POST'])
def recognize():

    if request.method == "POST":
        print("Receive image and is predicting it")

        data = request.get_json()
        imageBase64 = data['image']
        imgBytes = base64.b64decode(imageBase64)

        with open("temp.jpg", "wb") as temp:
            temp.write(imgBytes)

        image = cv2.imread('temp.jpg')
        image = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_prediction = np.reshape(image_gray, (28,28,1))

        #normalise

        image_prediction = (255 - image_prediction.astype('float')) / 255
        prediction = np.argmax(model.predict(np.array([image_prediction])), axis=-1)

        #running predict here
        return jsonify({
            'prediction': str(prediction[0]),
            'status':True
        })

    return render_template('/index.html')


if __name__ == '__main__':
    app.run(debug=True)