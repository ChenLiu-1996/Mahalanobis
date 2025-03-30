# Name To Be Determined

## Reproducing Results.

### Train with and without full-gradient
Using (CIFAR-10 + SimCLR + ResNet) as an example.
```
cd src
python main.py --dataset cifar10 --model resnet --learning-method simclr --random-seed 1 --full-grad
python main.py --dataset cifar10 --model resnet --learning-method simclr --random-seed 1
```


## Preparation

### Environment
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
conda create --name BetterCosine pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda activate BetterCosine
conda install scikit-image scikit-learn pillow matplotlib seaborn tqdm -c anaconda -y
python -m pip install tinyimagenet
python -m pip install natsort
python -m pip install phate
python -m pip install timm
```


### Dataset
#### Most datasets
Most datasets (MNIST, CIFAR-10, CIFAR-100, STL-10) can be directly downloaded via the torchvision API as you run the training code. However, for the following datasets, additional effort is required.

#### ImageNet data
NOTE: In order to download the images using wget, you need to first request access from http://image-net.org/download-images.
```
cd data/
mkdir imagenet && cd imagenet
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz

#### The following lines are instructions from Facebook Research. https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset.
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

```

