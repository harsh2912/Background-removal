from flask import Flask, jsonify, request
from demo import pred
from model import Model
import pickle
import numpy as np

Model.eval();

app = Flask(__name__)

def get_arr(arg):
    return np.array(arg)
    


def gen_trimap(trimap):
    trimap_im = get_arr(trimap)
    h, w = trimap_im.shape
    trimap = np.zeros((h, w, 2))
    trimap[trimap_im == 1, 1] = 1
    trimap[trimap_im == 0, 0] = 1
    return trimap


@app.route('/',methods=['POST'])
def matte():
    data = request.get_json()
    trimap = data.get('trimap')
    image = get_arr(data.get('image'))
    trimap = gen_trimap(trimap)
    fg,bg,alpha = pred(image,trimap,Model)
    return jsonify({'alpha':alpha.tolist()})
    

if __name__=='__main__':
    app.run()