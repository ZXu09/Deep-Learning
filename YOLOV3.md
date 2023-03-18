
1. [一、预测部分](#一预测部分)
2. [主干特征提取网络darknet53介绍](#1主干特征提取网络darknet53介绍)
    - [残差网络](#残差网络)
    - [darknet53](#darknet53)
3. [从特征获取预测结果](#2从特征获取预测结果)
    - [构建FPN特征金字塔进行加强特征提取](#构建fpn特征金字塔进行加强特征提取)
    - [利用YoloHead获得预测结果](#利用YoloHead获得预测结果)
4. [预测结果的解码](#3预测结果的解码)
    - [预测框位置计算](#预测框位置计算)
    - [对于每一个类进行判别](#对于每一个类进行判别)
5. [在原图上进行绘制](#4在原图上进行绘制)
***
1. [二、训练部分](#二训练部分)
    - [评价指标](#评价指标)
    - [梯度下降法](#梯度下降法)
2. [计算loss所需参数](#1计算loss所需参数)
3. [pred是什么](#2pred是什么)
4. [target是什么](#3target是什么)
5. [loss的计算过程](#4loss的计算过程)

生成目录：快捷键CTRL(CMD)+SHIFT+P，输入Markdown All in One: Create Table of Contents回车
# yolov3-pytorch
# 一、预测部分
## 1、主干特征提取网络darknet53介绍
输入416×416×3->进行下采样，宽高不断压缩，同时通道数不断扩张；若是进行上采样，宽高不断扩张，同时通道数不断压缩。

![darknet53](https://github.com/SZUZOUXu/yolov3-pytorch/blob/main/image/darknet53.jpg)

残差网络介绍：[Resnet50.md](https://github.com/SZUZOUXu/Deep-Learning/blob/main/Resnet50.md)

## darknet53
1. Darknet53的每一个卷积部分使用了特有的DarknetConv2D结构，每一次卷积的时候进行l2正则化，完成卷积后进行**BatchNormalization标准化与LeakyReLU**。加入这两个部分的目的是为了防止过拟合。  
普通的ReLU是将所有的负值都设为零，**Leaky ReLU**则是给所有**负值赋予一个非零斜率**。  
Relu的输入值为负的时候，输出始终为0，其一阶导数也始终为0，这样会导致神经元不能更新参数（Dead ReLU）。**Leaky ReLU**解决了Relu函数进入负区间后，导致神经元不学习的问题。

<div align=center>
<img src="https://github.com/SZUZOUXu/yolov3-pytorch/blob/main/image/Darknetconv2D_BN_Leaky.png"/>
</div>

2. Darknet53具有一个重要特点是使用了**残差网络Residual**，
- Darknet53中的残差卷积首先进行一次卷积核大小为3X3、步长为2的卷积，该卷积会压缩**输入进来的特征层的宽和高**。此时我们可以获得一个特征层，我们将该**特征层命名为layer**。
- 再对该特征层进行一次**1X1的卷积下降通道数**，然后利用一个**3x3卷积提取特征并且上升通道数**。把这个结果**加上layer**，此时我们便构成了**残差结构**。
- 没有池化层，通过步长为2的下采样实现。

### 池化和卷积下采样的区别
1. 卷积过程导致的图像变小是为了**提取特征**，控制了步进大小，**信息融合较好**。
2. 池化下采样是为了**降低特征的维度**，池化下采样**比较粗暴**，可能将有用的信息滤除掉。

- 416,416,32 -> 208,208,64
- 3×3卷积，步长为2：416,416,32 -> 208,208,64
- 1×1卷积降维：208,208,64->208,208,32
- 3×3卷积升维：208,208,32->208,208,64

代码如下位置："yolo3-pytorch\nets\darknet.py"

## 2、从特征获取预测结果
三个特征层的shape分别为(52,52,256)、(26,26,512)、(13,13,1024)。

从特征获取预测结果的过程可以分为两个部分，分别是：

- 构建**FPN特征金字塔**进行加强**特征提取**。
- 利用**Yolo Head**对三个有效特征层进行**预测**。

### 构建FPN特征金字塔进行加强特征提取
- **上采样UmSampling2d**：上采样使用的方式为上池化，即**元素复制扩充**的方法使得特征图尺寸扩大
- **拼接concat**：深层与浅层的特征图进行**拼接**，特征图按照通道维度直接进行拼接，例如8 * 8 * 16的特征图与8 * 8 * 16的特征图拼接后生成8 * 8 * 32的特征图
- **FPN(Feature Pyramid Networks)特征金字塔**：特征金字塔可以将不同shape的特征层进行**特征融合**，有利于**提取出更好的特征**。

### 为什么要采用FPN特征金字塔？
1. 卷积网络中，随着网络**深度的增加**，特征图的**尺寸越来越小**，语义信息也越来越抽象，**感受野变大，检测大物体**。
2. 浅层特征图的语义信息较少，目标位置相对比较准确，深层特征图的语义信息比较丰富，目标位置则比较粗略，导致小物体容易检测不到。
3. FPN的功能可以说是**融合了浅层到深层的特征图** ，从而**充分利用各个层次的特征**。

### 利用YoloHead获得预测结果
利用**FPN特征金字塔**，我们可以获得**三个加强特征->橙色框**，这三个加强特征的shape分别为(13,13,512)、(26,26,256)、(52,52,128)，然后我们利用这三个shape的特征层传入Yolo Head获得预测结果。

- Yolo Head本质上是一次3x3卷积加上一次1x1卷积，**3x3卷积的作用是特征整合，1x1卷积的作用是调整通道数**。

1. (13,13,75)->(13,13,3,25)
13×13的网格，**每个网格3个先验框**（预先标注在图片上，预测结果判断先验框内部有无物体，对先验框进行中心还有宽高的调整）
2. (13,13,3,25)->(13,13,3,20+1+4)
VOC数据集分20个类，1判断先验框内是否有物体，4代表先验框的调整参数（4个参数确定一个框的位置）
3. 最后的维度应该为**75 = 3x25**

如果使用的是coco训练集，类则为80种，最后的维度应该为**255 = 3x85**

### 总结
其实际情况就是，输入N张416x416的图片，在经过多层的运算后，会输出三个shape分别为(N,13,13,75)，(N,26,26,75)，(N,52,52,75)的数据

对应每个图分为**13x13、26x26、52x52的网格上3个先验框的位置**。

代码如下位置："yolo3-pytorch\nets\yolo.py"

## 3、预测结果的解码
由第二步我们可以获得三个特征层的预测结果，shape分别为：

- (N,13,13,75)
- (N,13,13,75)
- (N,13,13,75)

在这里我们简单了解一下每个有效特征层到底做了什么：(注意**大的特征图**由于**感受野较小**，同时特征包含位置信息丰富，适合检测**小物体**。)

- 每一个有效特征层将整个图片分成与其长宽对应的网格，如(N,13,13,75)的特征层就是将整个图像分成**13x13个网格**；
- 然后从**每个网格中心建立多个先验框**，这些框是网络预先设定好的框，网络的预测结果会**判断这些框内是否包含物体，以及这个物体的种类**。

由于每一个网格点都具有三个先验框，所以上述的预测结果可以reshape为：

- (N,13,13,3,25)
- (N,13,13,3,25)
- (N,13,13,3,25)
25可以拆分为4+1+20其中的4代表先验框的调整参数，1代表先验框内是否包含物体，20代表VOC数据集的种类

确切而言，4+1对应：**x_offset、y_offset、h和w、置信度**。

### 预测框位置计算：

- 先将**每个网格点加上它对应的x_offset和y_offset**，加完后的结果就是**预测框的中心**。
- 然后再利用**先验框和h、w结合计算出预测框的宽高**。这样就能得到整个预测框的位置了。

### 对于每一个类进行判别
- **得分排序**
取出**得分大于self.obj_threshold的框和得分**，也就是得分**满足confidence置信度**的预测框

- **非极大抑制筛选**
非极大抑制就是**选取得分最高**的那个框，接下来计算**当前框与其他的框的重合程度（iou），如果重合程度大于一定阈值就删除**（去除重叠的框）。  
若有多个目标物体，就是一个迭代的过程。

- **判断类别**
物体检测任务中可能一个物体有多个标签  
**logistic激活函数**来完成，这样就能预测每一个类别是/不是
代码如下位置："yolo3-pytorch\utils\utils_bbox.py"

## 4、在原图上进行绘制
- 通过第三步，我们可以获得预测框在原图上的位置，而且这些预测框都是经过筛选的。
- 这些筛选后的框可以直接绘制在图片上，就可以获得结果了。

### 处理长宽不同的图片->padding
对于很多分类、目标检测算法，输入的图片长宽是一样的，如224,224、416,416等。直接resize的话，图片就会失真。

但是我们可以采用如下的代码，使其用padding的方式不失真。
```Python
from PIL import Image
def letterbox_image(image, size):# size是想要的尺寸
    # 对图片进行resize，使图片不失真。在空缺的地方进行padding
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)# 取小边/原来的边（缩小系数），一般是缩小了尺寸（size<image.size）
    nw = int(iw*scale)//按缩小系数之后的边长，还保持原来的比例
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))# PIL.Image.new(mode, size, color)，用给定的模式和大小创建一个新图像，灰色填充
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))# //整数除法，向下取整（得到了应当复制过去的图像的左上角坐标）
    return new_image

img = Image.open("2007_000039.jpg")
new_image = letterbox_image(img,[416,416])
new_image.show()

```
# 二、训练部分
## 评价指标
- IOU：(A ∩ B)/(A U B),用于判断正例
- Precision（精度）：预测之中的正确率（挑出的西瓜中好瓜的概率）
- Recall（召回率）：好瓜有多少比例被挑出来了
- AP：PR曲线与x轴围成的面积，越接近1越好
- 预测框置信度confidence：(box 内存在对象的概率 * box 与该对象实际 box 的 IOU)
- mAP(mean Average Precision)即各类别AP的平均值  
IOU from 0.5 to 0.95 with a step size of 0.05，共计9个IOU（0.45/0.05），20个种类（VOC）  
计算IOU = 0.5作为confidence时的AP...->计算mAP  
高于阈值的边界框被视为正框，因此，置信度阈值越高，mAP 就越低，但我们对准确性更有信心。

optimizer优化器：sgd
## 梯度下降法
### 批量梯度下降法
如果使用梯度下降法(批量梯度下降法)，那么每次迭代过程中都要对**n个样本**进行**求梯度**，所以开销非常大。
### 随机梯度下降法（stochastic gradient descent，SGD）
随机梯度下降的思想就是随机采样**一个样本**来更新参数

随机梯度下降虽然提高了计算效率，降低了计算开销，但是由于每次迭代只随机选择一个样本，因此随机性比较大，所以下降过程中非常曲折。
### 小批量梯度下降法
可以选取一定数目的样本组成一个**小批量样本**，然后用这个小批量更新梯度

lr_decay_type 学习率下降：cos

weight_decay：权值衰减，可防止过拟合

### 自适应矩估计（Adaptive Moment Estimation，Adam）
SGD 低效的根本原因是，梯度的方向并没有指向最小值的方向。为了改正SGD的缺点，引入了Adam。

梯度下降速度快，但是容易在最优值附近震荡。
## 1、计算loss所需参数
在计算loss的时候，实际上是**pred和target**之间的对比：
- pred就是网络的预测结果。
- target就是网络的真实框情况。

## 2、pred是什么
对于yolo3的模型来说，网络最后输出的内容就是**三个特征层每个网格点对应的预测框及其种类**，即三个特征层分别对应着图片被分为不同size的网格后，**每个网格点上三个先验框对应的位置、置信度及其种类**。

输出层的shape分别为(13,13,75)，(26,26,75)，(52,52,75)，最后一个维度为75是因为是基于voc数据集的，它的类为20种，yolo3只有针对每一个特征层存在3个先验框，所以最后维度为3x25；
如果使用的是coco训练集，类则为80种，最后的维度应该为255 = 3x85，三个特征层的shape为(13,13,255)，(26,26,255)，(52,52,255)

现在的y_pre还是没有解码的，解码了之后才是真实图像上的情况。

## 3、target是什么
target就是**一个真实图像中，真实框的情况**。
第一个维度是batch_size，第二个维度是每一张图片里面真实框的**数量**，第三个维度内部是真实框的信息，包括**位置以及种类**。

## 4、loss的计算过程
拿到pred和target后，不可以简单的减一下作为对比，需要进行如下步骤。

判断真实框在图片中的位置，判断其**属于哪一个网格点去检测**。判断真实框和这个特征点的**哪个先验框重合程度最高**。计算该网格点应该有**怎么样的预测结果才能获得真实框**，与真实框重合度最高的**先验框被用于作为正样本**。

根据网络的预测结果获得**预测框**，计算预测框和所有真实框的**重合程度**，通过GIOU，如果重合程度大于一定门限，则将**该预测框对应的先验框忽略**。其余作为负样本。

最终损失由三个部分组成：
1. 正样本，编码后的**长宽与xy轴偏移量与预测值的差距**。
2. 正样本，预测结果中**置信度的值与1对比**；负样本，预测结果中**置信度的值与0对比**。
3. 实际存在的框，**种类预测结果与实际结果的对比**。