
from flask import Flask, request, render_template, jsonify
import base64
app = Flask(__name__)


import cv2
import numpy as np

import matplotlib.pyplot as plt


#get theta three four
theta = np.loadtxt("theta.txt") 

def sigmoid(s): #use formula in the logistic regression 
    return 1/(1 + np.exp(-s))

def findNumber(image, im_th, theta):
    contours, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(contour) for contour in contours]
    for rect in rects:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 6)
        length = int(rect[3]*1.6)
        pt1 = int(rect[1] + rect[3] // 2 - length //2)
        pt2 = int(rect[0] + rect[2] // 2 - length //2)

        roi = im_th[pt1:pt1+length, pt2:pt2+length]
        #resize roi
        roi = cv2.resize(roi,(28,28),interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))

        #input is an array of 28x28
        x = np.array((roi).reshape(1, 28*28))
        #adding ones
        ones = np.ones((x.shape[0],1))

        x = np.concatenate((x, ones), axis = 1)
        predict = sigmoid(np.dot(x, theta.T))

    return str(int(predict))

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
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0) 


        _, im_th = cv2.threshold(image_gray, 155, 255, cv2.THRESH_BINARY_INV)

        number = findNumber(image, im_th, theta)
        print(number)

        #running predict here
        return jsonify({
            'prediction': number,
            'status':True
        })

    return render_template('/index.html')


if __name__ == '__main__':
    app.run(debug=True)