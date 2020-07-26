import cv2
import numpy as np
import argparse
import requests
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_path',type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    img_path = args.image_path
    
    out_path = args.output_path
    img = cv2.imread(img_path)[:,:,::-1]
#     print(img.shape)
    response = requests.post('http://127.0.0.1:5000/',json={'image':img.tolist()})
    out = np.array(response.json()['output'])
    plt.imsave(out_path+'/out.jpg',out.astype('uint8'))