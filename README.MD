#Using VGG16 to classify cifar-10 by Transfer Learning

## Introduction
The original input size of VGG16 is 224\*224, but the CIFAR image size:32\*32. I am not going to resize the CIFAR image size to 224\*224. In order to do so, 
1) I need remove the fully connected layer of VGG16 and only reserve the conv layers.
2) append my own fully connected layers to the top.
3) Retain the VGG16 weight from change.
4) Train my own fc layers to classify the image.


In the conv3 sublayer, the size is not change. The pooling layer descrease the size in half. If the input size is 32*32, the output of pool5 will be 1*1

##Test result
### Fixed VGG16 conv layer
The valication accuracy has only about 60%.
