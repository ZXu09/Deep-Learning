
- [yolov3-pytorch](#yolov3-pytorch)
- [一、预测部分](#一预测部分)
  - [1、主干特征提取网络darknet53介绍](#1主干特征提取网络darknet53介绍)
  - [darknet53](#darknet53)
    - [池化和卷积下采样的区别](#池化和卷积下采样的区别)
  - [2、从特征获取预测结果](#2从特征获取预测结果)
    - [构建FPN特征金字塔进行加强特征提取](#构建fpn特征金字塔进行加强特征提取)
    - [为什么要采用FPN特征金字塔？](#为什么要采用fpn特征金字塔)
    - [利用YoloHead获得预测结果](#利用yolohead获得预测结果)
    - [总结](#总结)
  - [3、预测结果的解码](#3预测结果的解码)
    - [预测框位置计算：](#预测框位置计算)
    - [对于每一个类进行判别](#对于每一个类进行判别)
  - [在原图上进行绘制](#在原图上进行绘制)
    - [处理长宽不同的图片-\>padding](#处理长宽不同的图片-padding)
- [二、训练部分](#二训练部分)
  - [评价指标](#评价指标)
  - [loss求解](#loss求解)
    - [Bounding box prior先验框](#bounding-box-prior先验框)
    - [Bounding box prior样本划分](#bounding-box-prior样本划分)
    - [训练用例->Bounding box prior](#训练用例-bounding-box-prior)
    - [Bounding box](#bounding-box)
    - [种类-class confidence](#种类-class-confidence)
    - [置信度confidence](#置信度confidence)
    - [损失函数](#损失函数)


生成目录：快捷键CTRL(CMD)+SHIFT+P，输入Markdown All in One: Create Table of Contents回车
# yolov3-pytorch
### 为什么叫You Only Look Once
因为YOLOv3是**One-stage算法**，它在预测时只需要进行**一次前向传播直接回归物体的类别概率和位置坐标值**，使得**单一的网络就能够同时完成定位和分类**。运行速度明显快于具有可比性能的其他检测方法。
- **One-stage**：直接回归物体的类别概率和位置坐标值，单一的网络就能够同时完成定位和分类
- **Two stage**：首先产生候选区域（region proposals），先定位、后识别的任务。

### 回归问题、分类问题
YOLOv3 将对象检测问题构建为两步问题，首先识别边界框（回归问题），然后识别该对象的类（分类问题）。
- 回归算法用于连续型样本的预测
- 分类算法用于离散的类别标签

### YOLOV3的改进、优势
- 使用多尺度特征进行对象检测，优化了对小物体的检测效果。
- 改进了多个独立的Logistic regression分类器来取代softmax来预测类别分类，方便支持多标签物体。
- 当使用mAP50作为评估指标时，YOLOv3的表现非常惊人，在精确度相当的情况下，YOLOv3的速度是其他模型的3,4倍。

![YOLOV3 mAP-50](https://github.com/SZUZOUXu/Deep-Learning/blob/main/image/YOLOV3/YOLOV3%20map50.png)

### 改进YOLOv3
- 先验框的数量多且正负样本不均衡，实际项目中大部分物体都是差不多大的，或者说仅仅有特定的几种尺度，此时采用k-mean这一套流程就会出现：几乎一样大的物体被强制分到不同层去预测
- 由于大部分物体都是中等尺寸物体，会出现其余两个分支没有得到很好训练，或者说根本就没有训练，浪费网络。  
感受野和物体大小，基于网络输出层感受野，定义三个大概范围尺度分别进行k-means
- 输入数据->数据量增大、数据增强
- 不在每个级别对特征进行预测，而是首先重新缩放三个级别的特征，然后在每个级别自适应组合，然后对这些新功能执行预测/检测。

# 一、预测部分
## 1、主干特征提取网络darknet53介绍
输入416×416×3->进行下采样，宽高不断压缩，同时通道数不断扩张；若是进行上采样，宽高不断扩张，同时通道数不断压缩。

### 数据增强
数据增强其实就是让图片变得更加多样
- 对图像进行缩放并进行长和宽的扭曲
- 对图像进行翻转
- 对图像进行色域扭曲

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
用stride=2的conv降采样的卷积神经网络效果与使用pooling降采样的卷积神经网络效果相当；卷积神经网络小的时候，使用pooling降采样效果可能更好，卷积神经网络大的时候，使用stride=2的conv降采样效果可能更好。  
总结：pooling提供了一种非线性，当网络很深的时候，多层叠加的conv可以学到pooling所能提供的非线性

### 不采用全连接层的原因？
采用了卷积和anchor boxes来预测边界框。

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

## 在原图上进行绘制
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
训练过程：
- 三个有效特征层循环计算损失。
- 反向传播进行训练。
## 评价指标
- IOU：(A ∩ B)/(A U B),用于判断正例
- Precision（精度）：预测之中的正确率（挑出的西瓜中好瓜的概率）
- Recall（召回率）：好瓜有多少比例被挑出来了
- AP：PR曲线与x轴围成的面积，越接近1越好
- 预测框置信度confidence：(box内存在对象的概率 * box 与该对象实际box的IOU)
- mAP(mean Average Precision)即各类别AP的平均值  
IOU from 0.5 to 0.95 with a step size of 0.05，共计9个IOU（0.45/0.05），20个种类（VOC）  
计算IOU = 0.5作为confidence时的AP...->计算mAP  
高于阈值的边界框被视为正框，因此，置信度阈值越高，mAP就越低，但我们对准确性更有信心。
- optimizer优化器：sgd

## loss求解
### 计算loss的pred、target
在计算loss的时候，实际上是**pred和target**之间的对比：
- pred就是网络的预测结果。**三个特征层每个网格点对应的预测框及其种类**
- target就是网络的真实框情况。**真实框的x，y，w，h、种类**

![loss求解](https://github.com/SZUZOUXu/Deep-Learning/blob/main/image/loss%E6%B1%82%E8%A7%A3.jpg)

### Bounding box prior先验框
bounding box prior如图所示：

![YOLOV3 anchor box](https://github.com/SZUZOUXu/Deep-Learning/blob/main/image/YOLOV3%20anchor%20box.png)

每个尺度的特征图会预测出3个bounding box prior, 而bounding box prior的大小则采用**K-means进行聚类分析**，在训练集中所有样本的真实框中聚类出**具有代表性形状的宽和高**。  
**对每个真实框分配一个IOU最高的先验框（1对1）**，其objectness score = 1若一个真实框分配多个IOU，会损失计算的物体存在的概率（objectness）

K-means见：[K-means聚类](https://github.com/SZUZOUXu/Deep-Learning/blob/main/K-mean%E8%81%9A%E7%B1%BB.md)

### Bounding box prior 的作用：
**降低模型学习难度，模型训练更加稳定，获得更高的precision。**  
因为模型不会直接预测出预测框，而是通过先验框以及一个转换函数T得到预测框。使得预测框更有针对性。  

### Bounding box prior样本划分
**将先验框与真实框计算IOU**  
先验框一共分为三种情况：正例（positive）、负例（negative）、忽略样例（ignore）。
- **正例**：与真实边界框的IOU为最大值，且高于阈值 0.5的bounding box；类别标签对应类别为1，其余为0；置信度标签为1。**产生所有损失**
- **忽略样例**：与真实边界框的IOU 非最大值，但仍高于阈值0.5的bounding box 则不产生cost。**不产生任何损失**
- **负例**：低于阈值。**只产生置信度损失**  

### 训练的时候为什么需要进行正负样本筛选？
在目标检测中不能将所有的预测框都进入损失函数进行计算，主要原因是框太多，参数量太大，因此需要先将正负样本选择出来，再进行损失函数的计算。

### 为什么有忽略样例？
由于Yolov3使用了多尺度特征图，**不同尺度的特征图之间会有重合检测部分**。  
比如有一个真实物体，在训练时被分配到的先验框是特征图1的第三个box，IOU达0.98，此时恰好特征图2的第一个box与该ground truth的IOU达0.95，也检测到了该ground truth。
**如果给IOU高的物体的置信度强行打0的标签，网络学习效果会不理想。**

### 训练用例->Bounding box prior
每个网格对应3个先验框，当作样本进行训练，如图所示：

![YOLOV3 anchors](https://github.com/SZUZOUXu/Deep-Learning/blob/main/image/YOLOV3%20anchors.jpg)

### Bounding box
由三个特征层的输出结果和Bounding box prior可以计算得到最终的预测框

![YOLOV3 Bounding box](https://github.com/SZUZOUXu/Deep-Learning/blob/main/image/YOLOV3%20anchor%20box.png)

其中：
$𝑏_{𝑥}$ 和 $𝑏_{𝑦}$ 是边界框的中心坐标，𝜎(𝑥)为sigmoid函数， $c_{x}$ 和 $𝑐_{𝑦}$ 分别为方格左上角点相对整张图片的坐标。
$𝑝_{𝑤}$ 和 $𝑝_{ℎ}$ 为anchor box的宽和高， $𝑡_{w}$ 和 $𝑡_{ℎ}$ 为边界框直接预测出的宽和高， $𝑏_{𝑤}$ 和 $𝑏_{ℎ}$ 为转换后预测的实际宽和高。  
网络为每个bounding box预测4个值 $t_{𝑥}$ 、 $t_{𝑦}$ 、 $t_{w}$ 和 $𝑡_{ℎ}$

特征层中的每一个方格（grid cell）都会预测3个边界框（bounding box），每个边界框都会预测三个东西：  
- 每个框的位置 $𝑏_{𝑥}$ 和 $𝑏_{𝑦}$ 、 $𝑝_{𝑤}$ 和 $𝑝_{ℎ}$ 
- 框内物体的置信度confidence(0 ~ 1)
- 框内物体的种类（VOC数据集共20种、COCO数据集共80种）(0 ~ 1)

在训练中我们挑选哪个bounding box的准则是选择预测的box与ground truth box的IOU最大的bounding box做为最优的box，
但是在预测中并没有ground truth box，怎么才能挑选最优的bounding box呢？这就需要另外的参数了，那就是下面要说到的置信度。

### 学习目标是tx，ty，tw，th 偏移量而不是直接学习bx，by，bw，bh呢？
bx，by，bw，bh的数值大小和 objectness score 以及 class probilities 差太多了，会**给训练带来困难**。

### 为什么要用指数？
公式中多加了exp操作，应该是为了保证缩放比大于 0，不然在优化的时候会多一个 $𝑡_{w} > 0$ 和 $𝑡_{ℎ} > 0$ 的约束，这时候**SGD这种无约束求极值算法**是用不了的。

### 种类-class confidence
- 在置信度表示**当前box有对象**的前提下进行计算；（objectness score = 1）
- 实现多标签分类：**Logistic regression分类器**：  
实现多标签分类就需要用Logistic regression分类器来**对每个类别都进行二分类**。Logistic regression分类器主要用到了**sigmoid函数**，它可以把输出约束在0到1，如果某一特征图的输出经过该函数处理后的值**大于设定阈值**，那么就认定该目标框所对应的**目标属于该类**。

### 置信度confidence
最终显示的概率为**物体存在的概率**（objectness）和物体**分类的置信度**（class confidence）**相乘**。
- **物体存在的概率（objectness）**：(20 + 1 + 4)，输出的就是物体存在的概率
- 物体**分类的置信度**：(20 + 1 + 4)，输出的就是分类的条件概率
如图所示：

![YOLOV3 Bounding box](https://github.com/SZUZOUXu/Deep-Learning/blob/main/image/YOLOV3%20confidence.png)

**相当于最终显示的概率是class confidence score**

### 损失函数
对**位置、种类、置信度**进行学习；  
x、y、w、h使用**MSE**（均方误差）作为损失函数，置信度、类别标签由于是0，1二分类，所以使用**交叉熵**作为损失函数。  
- 预测框置信度： (box 内存在对象的概率 * box 与该对象实际 box 的 IOU)
- 一个预测框的置信度 (Confidence) 代表了是否包含对象且位置正确的准确度.
损失函数如图所示：

![YOLOV3损失函数](https://github.com/SZUZOUXu/Deep-Learning/blob/main/image/YOLOV3%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.png)

