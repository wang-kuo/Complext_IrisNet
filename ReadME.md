This is an unofficial pytorch implementation of ["Complex-valued Iris Recognition Network"](https://arxiv.org/abs/2011.11198).

## Requirements
Please install the following dependency in this project.
```bash
opencv-python
matplotlib
numpy
scipy
scikit
pytorch==1.11.0
torchvision==0.13.0
cplxmodule
```

## Training
Please directly run `python train.py` to start training and all the snapshot are saved in [snapshot](./snapshot/).

## Testing
Please directly run `python test.py` to start training and results will be plotted in ROC.png.