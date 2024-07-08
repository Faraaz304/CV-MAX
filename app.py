from flask import Flask,request,render_template
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.FaceDetectionModule import FaceDetector

import numpy as np

app = Flask(__name__)


def hex_to_bgr(hex_value):
    # Remove the hash (#) at the start if it's there
    hex_value = hex_value.lstrip('#')
    
    # Convert the hex value to an integer tuple
    rgb_tuple = tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))
    
    # Rearrange the tuple to BGR
    bgr_tuple = (rgb_tuple[2], rgb_tuple[1], rgb_tuple[0])
    
    return bgr_tuple


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/background_remover' , methods =['POST','GET'])
def baground_remover():
     if request.method =='POST':
        img =request.files['image']
        img.save('static/img.png')
        temp = cv2.imread('static/img.png')


        rem = SelfiSegmentation()
        colour =  request.form.get('col')


        a= rem.removeBG(temp , hex_to_bgr(colour) ,cutThreshold=0.65)

        a =cv2.imwrite("static/output.png" ,a)
        




     return render_template('baground_remover.html')

@app.route('/face_mesh',methods =['POST','GET'])
def face_mesh():
    if request.method =='POST':
        img =request.files['image']
        img.save('static/img.png')
        temp = cv2.imread('static/img.png')
        detector =FaceMeshDetector()
        result, faces = detector.findFaceMesh(temp)
        cv2.imwrite('static/output.png',result)


    return render_template('face_mesh.html')


@app.route('/face_detector' , methods =['POST','GET'])
def face_detector():
     if request.method =='POST':
        img =request.files['image']
        img.save('static/img.png')
        temp = cv2.imread('static/img.png')

        detector  = FaceDetector()
        result, box = detector.findFaces(temp)

        cv2.imwrite('static/output.png', result)
        
     return render_template('face_detection.html')




if __name__  == '__main__':
    app.run(debug=True)