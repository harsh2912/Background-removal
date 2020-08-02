
# Bakcground Removal


<p align="center">
  <img src="./examples/example_results.png" width="840" title="FBA_matting results"/>
</p>

## Requirements
GPU memory >= 11GB for inference on Adobe Composition-1K testing set, more generally for resolutions above 1920x1080.

#### Packages:
- torch >= 1.4
- numpy
- opencv-python
#### Additional Packages for jupyter notebook
- matplotlib



## Prediction 
There is a script `inference.py` which gives the background subtracted from the provided image

## Citation

Original github repository of FBA_matting : [link](https://github.com/MarcoForte/FBA_Matting)
```
@article{forte2020fbamatting,
  title   = {F, B, Alpha Matting},
  author  = {Marco Forte and François Pitié},
  journal = {CoRR},
  volume  = {abs/2003.07711},
  year    = {2020},
}
```

