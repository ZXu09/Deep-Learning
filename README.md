# 下载和使用深度学习库
## 下载和打开深度学习库
- git clone在对应文件夹，并用VS code打开
- 阅读requirement.txt文件，安装对应Python库
以Yolov5为例：
```Python
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install（打开对应requirement.txt的文件目录下进行操作）
```
## 使用深度学习库
### 训练
- 阅读train相关的README信息
以Yolov5为例：在对应文件的命令行中使用
```Python
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
```
### 预测
-关键字predict/Inference/detect

## 分解与学习深度学习库
### 深度学习库的组成
一般来讲，深度学习库的功能包括两个部分：
1. 训练模型
- 在训练模型时，需要考虑模型本身，训练参数，数据加载与损失函数。
2. 利用模型进行预测。
- 在预测模型时，需要考虑模型本身，数据加载，预测后处理。

综合起来，一个能用的深度学习库需要包含如下5个部分：
- 模型本身，训练参数，数据加载，损失函数，预测后处理。

### 一、模型本身
- 一般来讲，模型本身在仓库中的名字是net（网络）或者model（模型）。
以Yolov5为例，在model文件夹下：
- common.py 一些小模块的合集，即结构模块
- experiemtal.py 一些实验性质的代码
- tf.py tensorflow
- yolo.py 构建模型
YoloV5的库通过Yaml文件进行模型的构建

### 二、训练参数
训练参数一般伴随着训练文件，因此一般在train.py文件里面，每一个库指定参数的方式不同，有些喜欢通过yaml文件指定，有些喜欢通过cfg文件指定，有些甚至通过py文件指定，都不一样，这个需要参考每一个库的组成去分析。但大多数库都可以在train.py文件中找到蛛丝马迹。

yolov5通过argparse指定参数。argparse是python自带的命令行参数解析包，可以用来方便地读取命令行参数。结合yolov5训练的指令可知，yolov5利用argparse通过命令行获取参数。
```Python
python train.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
```
- yolov5n.yaml 模型参数
- coco.yaml 数据参数
- hyps的yaml文件 训练参数，有关数据增强，学习率(lr,learning)，损失函数等。

###三、数据加载
- 数据加载分为两部分，一部分是训练的数据加载，另一部分是预测的数据加载。
1. 训练部分
训练的数据加载其实是非常重要的，直接关系到模型的训练，监督模型在训练时加载的数据一般分为两部分，一部分是输入变量，通常是图片；另一部分是标签，在目标检测中就是图片对应的框的坐标，在语义分割中就是每个像素点的种类。

一般来讲，数据加载部分，模型本身在仓库中的名字是data、datasets或者dataloader。

- 初次看yolov5的库，可能会以为数据加载部分的内容在data文件夹中，点进去会发现，其实data都是数据集下载相关的内容，这种判断错误是正常的，毕竟这属于相似概念了。
- 实际的数据集加载文件，datasets文件在utils文件夹中。
简单翻一下datasets，可以知道该文件通过create_dataloader函数构建文件加载器，然后通过LoadImagesAndLabels这个文件加载器的类来获取图片与标签文件。
