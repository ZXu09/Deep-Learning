# 使用深度学习库
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

### 三、数据加载
- 数据加载分为两部分，一部分是训练的数据加载，另一部分是预测的数据加载。
1. 训练部分
训练的数据加载其实是非常重要的，直接关系到模型的训练，监督模型在训练时加载的数据一般分为两部分，一部分是输入变量，通常是图片；另一部分是标签，在目标检测中就是图片对应的框的坐标，在语义分割中就是每个像素点的种类。

一般来讲，数据加载部分，模型本身在仓库中的名字是data、datasets或者dataloader。

- 初次看yolov5的库，可能会以为数据加载部分的内容在data文件夹中，点进去会发现，其实data都是数据集下载相关的内容，这种判断错误是正常的，毕竟这属于相似概念了。
- 实际的数据集加载文件，dataloaders文件在utils文件夹中。
简单翻一下dataloaders.py，可以知道该文件通过create_dataloader函数构建文件加载器，然后通过LoadImagesAndLabels这个文件加载器的类来获取图片与标签文件。
```Python
def create_dataloader(path,imgsz,batch_size,stride,single_cls=False,hyp=None,augment=False,cache=False,pad=0.0,
                      rect=False,rank=-1,workers=8,image_weights=False,quad=False,prefix='',shuffle=False,seed=0):
```
然后在这个LoadImagesAndLabels中，算法会进行数据增强、数据预处理等操作，最终返回输入图片与标签。
```Python
class LoadImagesAndLabels(Dataset):
```
2. 预测部分
- 预测的数据加载和训练的数据加载相比，少了数据增强与标签处理的部分，因此会相对简单一些，主要是对输入图片进行预处理。
- 既然是预测部分的数据预处理，我们需要从预测文件开始寻找。
在detect.py预测文件中可以发现，YoloV5通过文件加载器的方式获得预测的图片文件，在文件加载器中，我们会对图片文件进行预处理，比如resize到一定的大小，进行图片的归一化等
```Python
if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * 
```
在utils文件夹中的datasets文件中的LoadImages：
```Python
lass LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

```
### 四、损失函数
一般来讲，损失函数在仓库中的名字是Loss（损失），Loss函数是模型优化的目标，在训练过程中Loss理论上是要被越优化越小的。

结合train.py调用的函数来看，可以很容易发现，yolov5计算损失时，调用的是ComputeLoss类。
```Python
compute_loss=ComputeLoss(model)//train.py
```
进一步定位Loss的计算。
```Python
class ComputeLoss://loss.py
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
```
Loss组成的话，每个仓库有每个仓库不同的组成方式，因此解析的难度是非常大的，特别是在目标检测中，正样本的选取方式多样，很难直接对Loss有个整体的认知，想要进一步了解Loss的工作，通常要对损失进行一行、一行的分析。

### 五、预测后处理
预测的后处理主要包括了预测结果的解码与预测图片的可视化。既然是预测部分的后处理，我们需要从预测文件开始寻找。
```Python
//在预测Inference之后进行，进行了非极大抑制，然后进行了图片的绘制与可视化。
 # NMS
with dt[2]:
     pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
# Second-stage classifier (optional)
# pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

# Process predictions
for i, det in enumerate(pred):  # per image
    seen += 1
if webcam:  # batch_size >= 1
    p, im0, frame = path[i], im0s[i].copy(), dataset.count
    s += f'{i}: '
else:
    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
```
## 分解与学习深度学习库
### 修改目标定位
在修改仓库前，需要根据所需要修改的功能对需要修改的目标进行定位，比如：
1. 想要对网络结构进行改进，那么就定位到模型本身，然后修改网络结构的特定部分。
2. 想要对训练参数进行改进，那么就定位到训练参数，然后查找各个参数的作用，进一步进行修改。
3. 想要对数据增强进行改进，那么就定位到数据加载，然后分解其中数据增强的部分，修改数据增强的过程。
4. 想要对损失函数进行改进，那么就定位到损失函数，然后分解其中的回归部分、分类部分等，进一步对细节进行修改。
5. 想要对预测结果进行改进，那么就定位到预测后处理，然后根据自身需求，分解每个输出的作用，进一步修改。
