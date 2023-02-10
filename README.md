# 下载和使用深度学习库
## 下载和打开
- git clone在对应文件夹，并用VS code打开
- 阅读requirement.txt文件，安装对应Python库
以Yolov5为例：
```Python
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install（打开对应requirement.txt的文件目录下进行操作）
```
## 使用
### train
- 阅读train相关的README信息
以Yolov5为例：在对应文件的命令行中使用
```Python
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
```
