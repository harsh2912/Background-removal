from flask import Flask, jsonify,request,Response
from segmentation import Model,Preprocessing
# from detectron_seg import Model,Preprocessing
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

def get_arr(arg):
    return np.array(arg).astype('uint8')

seg_model = Model()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

preprocessor = Preprocessing(kernel)


@app.route('/',methods=['POST'])
def gen_trimap():
    data = request.get_json()
    image = get_arr(data['image'])
    output = seg_model.get_seg_output(image)
    if len(output) == 0:
        return(Response(status=406))
    masks = np.array([mask.cpu().numpy() for mask,classes in output])
    trimap = preprocessor.get_trimap(masks)
    return jsonify({'trimap':trimap.tolist()})#,'masks':masks.tolist()})

if __name__=='__main__':
    app.run(debug=True,threaded=True)