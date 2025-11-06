# DFD-Net: Dynamic Feature Differentiation Network for Multi-modal Brain Tumor Segmentation
## Usage
### Data Preparation
Please download BraTS 2020 data according to https://www.med.upenn.edu/cbica/brats2020/data.html.
### Training
#### Training on the entire BraTS training set
```bash
python train.py --model DEDNet --mixed --trainset
```
#### Breakpoint continuation for training
```bash
python train.py --model DEDNet --mixed --trainset --cp checkpoint
```
### Inference
```bash
python test.py --model DEDNet --tta --labels --post_process --cp checkpoint
```
