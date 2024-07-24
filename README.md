# Eglcr
Eglcr
# EGLCR: Eglcr: Edge Structure Guidance and Scale Adaptive Attention for Iterative Stereo Matching


## Demos


### Create a virtual environment and activate it.

```
conda create -n EGLCR python=3.8
conda activate EGLCR
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c nvidia
conda activate EGLCR

pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install timm==0.5.4
```

You can demo a trained model on pairs of images, run
```
conda activate EGLCR
python demo.py --restore_ckpt ./checkpoint/pretrain.pth
```
