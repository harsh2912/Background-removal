from flask import Flask, jsonify,request
from segmentation import Model,Preprocessing
import cv2
import numpy as np

app = Flask(__name__)

def get_arr(arg):
    return np.array(arg).astype('uint8')

seg_model = Model()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

preprocessor = Preprocessing(kernel,10)


@app.route('/',methods=['POST'])
def gen_trimap():
    data = request.get_json()
    image = get_arr(data.get('image',None))
    output = seg_model.get_seg_output(image)
    masks = np.array([mask.cpu().numpy() for mask,classes in output])
    trimap = preprocessor.get_trimap(masks)
    return jsonify({'trimap':trimap.tolist(),'masks':masks.tolist()})

if __name__=='__main__':
    app.run()