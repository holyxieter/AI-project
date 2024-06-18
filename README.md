# AI-project
BJTU 软件学院 AI实践项目

### 项目文档

#### 项目概述

本项目在MNIST和CIFAR-10两个数据集上分别实现了五种经典的卷积神经网络架构：LeNet、AlexNet、VGGNet、ResNet和GoogLeNet。
针对不同的数据集，网络架构进行了适当调整，以确保模型在每个数据集上的性能表现。

#### 项目结构

```angular2html
mnist_project
├─load_data.py
├─train.py
├─grid_search.py
│
├─data
│  └─MNIST
│
└─model
   ├─alexnet.py
   │ googlenet.py
   │ lenet.py
   │ resnet.py
   └─vggnet.py
```
mnist_project 项目结构如上所示，包含以下内容：
- load_data.py: 用于数据预处理和加载的脚本。
- train.py: 负责模型的训练和评估的脚本。
- grid_search.py：负责模型网格调参的脚本。
- data: 存放 MNIST 数据集的子文件夹。
- model: 存放模型定义文件的文件夹：
    - alexnet.py: AlexNet 模型定义。
    - googlenet.py: GoogleNet 模型定义。
    - lenet.py: LeNet 模型定义。
    - resnet.py: ResNet 模型定义。
    - vggnet.py: VGGNet 模型定义。

```angular2html
cifar_project
├─load_data.py
├─train.py
├─grid_search.py
│
├─cifar-10-batches-py
│  ├─test
│  └─train
│
└─model
   ├─alexnet.py
   ├─googlenet.py
   ├─lenet.py
   ├─resnet.py
   └─vggnet.py
```
cifar_project 项目结构如上所示，包含以下内容：
- load_data.py: 用于数据预处理和加载的脚本。
- train.py: 负责模型的训练和评估的脚本。
- grid_search.py：负责模型网格调参的脚本。
- cifar-10-batches-py: 存放 CIFAR-10 数据集的子文件夹，分成训练集和测试集。
- model: 存放模型定义文件的文件夹：
    - alexnet.py: AlexNet 模型定义。
    - googlenet.py: GoogleNet 模型定义。
    - lenet.py: LeNet 模型定义。
    - resnet.py: ResNet 模型定义。
    - vggnet.py: VGGNet 模型定义。

#### 项目运行说明
- 请参考项目的requirements.txt文件，确保已安装项目所需的所有库依赖。
- 请自行下载 MNIST 和 CIFAR-10 数据集，并将其保存在对应的路径下。
- 运行项目文件夹中的train.py文件即可启动训练。
- 数据预处理等请在load_data.py文件中进行修改。
- 模型训练的超参数等请在train.py文件中进行调整。
- 模型架构请在相应的模型定义文件中进行修改。
