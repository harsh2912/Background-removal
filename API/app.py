from flask import Flask, jsonify,request
import sys
sys.path.append('..')
from networks.models import build_model
from matting import pred
import requests
import numpy as np

app = Flask(__name__)

def get_array(arg):
    return np.array(arg).astype('uint8')

class Matting_Args:
    def __init__(self):
        self.encoder = 'resnet50_GN_WS'
        self.decoder = 'fba_decoder'
        self.weights = '../FBA.pth'
        
args = Matting_Args()

matting_model = build_model(args)
matting_model.eval();

@app.route('/',methods=['POST'])
def extraction():
    data = request.get_json()
    image = get_array(data.get('image',None))/255.0
#     print(np.unique(image))
    response = requests.post('http://127.0.0.1:3001/',json = data)
    trimap = get_array(response.json()['trimap'])
    fg, bg, alpha = pred(image,trimap,matting_model)
    
    return jsonify({'output':((alpha[...,None]*image)*255.0).tolist()})


if __name__ == '__main__':
    app.run()
    