from flask import Flask, jsonify,request,url_for,redirect
import sys
sys.path.append('..')
from networks.models import build_model
from matting import pred
import requests
import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def get_array(arg):
    return np.array(arg).astype('uint8')

class Matting_Args:
    def __init__(self):
        self.encoder = 'resnet50_GN_WS'
        self.decoder = 'fba_decoder'
        self.weights = '../models/FBA.pth'
        
args = Matting_Args()

matting_model = build_model(args)
matting_model.eval();

def get_response(new_bg,data,image):
    response = requests.post('http://127.0.0.1:3000/',json = data)
    if response.status_code == 406:
        buffer = BytesIO()
        Image.fromarray(image).save(buffer,format='png')
        img_str = base64.b64encode(buffer.getvalue()).decode()
#         print(img_str)
        return {'output':img_str}
    
    h,w,_ = image.shape
    trimap = get_array(response.json()['trimap'])
    fg, bg, alpha = pred(image/255.0,trimap,matting_model)
    combined = ((alpha[...,None]*image)).astype('uint8') + ((1-alpha)[...,None]*cv2.resize(new_bg,(w,h))).astype('uint8')
    buffer = BytesIO()
    Image.fromarray(combined).save(buffer,format='png')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return {'output':img_str}

@app.route('/with_bg',methods=['POST'])
def extraction():
    data = request.get_json()
    new_bg = get_array(data.get('bg'))
    return get_response(new_bg,data)

@app.route('/',methods=["POST"])
def extraction_without_bg():
    data = request.files['file']
    img = np.array(Image.open(data.stream).convert(mode='RGB'))
#     print(img.shape)
    data = {'image':img.tolist()}
    new_bg = cv2.imread('1.jpg')[:,:,::-1]
    return get_response(new_bg,data,img)
        
    
if __name__ == '__main__':
    app.run(debug=True,threaded=True)
    