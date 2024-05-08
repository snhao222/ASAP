

<div align="center">
    <img src="./img/ASAP LOGO.png" width="300px"></img>
</div>
<p align="center">
    <img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-%2300CC99">
    <img alt="Static Badge" src="https://img.shields.io/badge/version-1.0.0-%23FFCC66">
    <img alt="Static Badge" src="https://img.shields.io/badge/email-sunhaosn%40nuaa.edu.cn-%236699FF">
</p>
<p align="center">
  <a href="README.md">English</a> |
  <a href="https://">简体中文</a>
</p>


<br>

本仓库包含IEEE Transactions on Mobile Computing（TMC）论文 “All Sky Autonomous Computing in UAV swarm” 的系统源代码。

ASAP (<u>A</u>ll <u>S</u>ky <u>A</u>utonomous com<u>P</u>uting) 是一套为无人机群设计的协同计算系统，其设计核心为在集群中动态地“借用”附近无人机的能力，实现机上大量传感载荷数据的实时处理。

<p align="center">
  <a href="#highlights">系统特色</a> •
  <a href="#technical details">技术细节</a> •
  <a href="#requirements">软件需求</a> •
  <a href="#usage">如何使用</a> •
  <a href="#contributing">帮助和建议</a> •
  <a href="#license">许可证</a> •
  <a href="#citation">引用本工作</a>
</p>

:star: 如果你喜欢这个项目，快给这个仓库点一个 ‘star’ 吧  : ) 

<h2 id="highlights">✨ 系统特色</h2>

<div style="text-align: left;">  
    <img align="right" src="./img/Application case.png" width="550px" style="float: right; margin-left: 10px;">
<p><ul><li>无人机集群原生的协同计算架构;</li></ul>
<ul><li>高效的弹性调度器，可实现高效、稳定的协同计算;</li></ul>
<ul><li>准确的轻量级深度学习推理性能预测器;</li></ul>
<ul><li>机间推理数据自适应压缩器，可根据网络情况自适应地减小传输数据量;</li></ul>
<ul><li>机间自主发现和接入控制。</li></ul></p>
<h4>⚡ 无人机群的杀手级应用：地震搜救</h4>
    <p align= "justify">由于基站通信链路可能中断（线路➍），现有的中继无人机辅助方法将大量现场图像传输回地面站，并以<b>耗时的方式</b>进行进一步处理（线路➋），或使用<b>低精度</b>的压缩深度学习模型分析机载图像（线路➌）。与之不同的是，ASAP提出在传输过程中同时地进行协同数据处理（线路➊），这可以将<b>高精度</b>的幸存者信息回传到地面站，并以<b>高效通信</b>的方式进行<b>快速决策</b>。
  </p>
</div>



<h2 id="technical details">📋 技术细节</h2>

<div align="center">
    <img src="./img/System overview.png" width="780px"></img>
</div>
ASAP的目标是在无人机群中**进行高效可靠的协同计算**，通过传输有价值的数据结果而不是原始数据来降低空地通信负载。

本系统由三个模块组成，即弹性高效任务调度器（Elastic Efficient Task Scheduler）、轻量级准确深度学习推理性能预测器（Lightweight Accurate DL Inference Performance Predictor）和无人机间推理数据自适应压缩器（Adaptive Inter-UAV Inference Data Compressor）。首先，当有任务需要处理时，即深度学习模型和数据序列，弹性高效任务调度器将任务分配给无人机群及其内部的无人机，以进行高效的协同计算。此外，当无人机状态发生变化时，例如一些无人机连接不可达或重新加入集群，任务调度器可以更新分配方6案。其次，任务的调度是在深度学习推理性能预测器的帮助下进行的，该预测器可以评估不同无人机上各种模型和数据块的推理时间。最后，无人机间传输的中间数据将由自适应压缩器进一步压缩，以节省无人机间的通信资源。

<details> <summary><b>系统模块设计</b>（展开以显示详细内容）</summary>
 本系统的模块组成如下图所示。工作将系统模块分为四类：控制模块（Control）、管理模块（Manage）、通信模块（Communication）和计算模块（Computing）。控制模块负责任务调度和数据分配，管理模块负责数据组织和维护，通信模块是无人机之间通信管理和模块间信息传输的中间件，计算模块处理与计算相关的工作。值得注意的是，模块间不直接交换信息和数据，而是通过内部通信模块，统一的通信接口能够确保模块之间的有序通信。
 <br>
 <br>
 <div align="center">
    <img src="./img/System composition.png" width="780px"></img>
 </div>
 <br>
 下图展示了主要模块间的关系。集群中的每个无人机都部署了一个性能测量模块，用于测量无人机的通信和计算性能。无人机的运行时信息由簇头的状态管理模块收集，以进行弹性任务调度。模型分割策略由发布任务的无人机上的任务调度器生成并将其发送给其他簇头无人机，每个簇头无人机上的任务调度器负责生成数据分割策略并将其发送给集群中的簇成员无人机。数据管理模块负责控制数据的流动，包括将数据块分配给计算模块进行本地计算，并整合簇成员返回的计算结果。
 <br>
 <div align="center">
    <img src="./img/System architecture.png" width="650px"></img>
 </div>
</details>
<details> <summary><b>工程部署技巧</b>（展开以显示详细内容）</summary>
<ul><li><b>数据分割中的 Padding</b></li></ul>
深度学习模型中的一些层在特征图周围填充零元素，以提升对边界信息的感知能力。然而，如下图所示，数据分割后的填充操作给特征图引入了一些无用信息，这会影响深度学习模型的感知能力。为了确保数据分割后的信息感知能力，我们为每个无人机的深度学习模型层自动在线生成了不同的Padding策略。详细地说，当无人机收到任务时，模型管理模块中的模型生成器将使用映射范围转换（MRT）算法计算每层的中间数据范围，并将该数据范围与每层的范围边界进行比较。若数据范围超过边界，相应侧的Padding将被置为零，并且这个过程将重复进行，直到整个模型被迭代遍历。
<div align="center">
    <img src="./img/Padding in data partition.png" width="300px"></img>
</div>
<ul><li><b>无人机接入技术</b></li></ul>
在本系统中，任何加入群集的无人机都可以被其他无人机自动发现，并相互交换身份信息。该无人机接入功能基于SSDP协议（简单服务发现协议），该协议允许设备在一个保留地址（239.255.255.250:1900）上广播消息。如下图所示，有两种消息被用于发现加入的无人机。<br>
1) M-SEARCH消息：此消息包含主机地址和唯一的服务名称（简称“usn”），以及集群信息和无人机编号。<br>
2) NOTIFY消息：此消息包含location信息（即无人机本地地址）和带有无人机信息的通知类型（简称“nt”）。
<br>当无人机加入集群时，将会周期性地广播M-SEARCH消息，相关无人机将以单播模式用NOTIFY消息进行响应。值得注意的是，无人机间的发现不仅发生在无人机簇中，不同簇的簇头之间也会互相接入。
<div align="center">
    <img src="./img/Access technique.png" width="350px"></img>
</div>
<ul><li><b>无人机状态监测</b></li></ul>
在弹性调度上，能够及时检测到无人机状态的变化至关重要。因此，本系统部署了一个状态管理模块来管理无人机的连接状态。在进行协同计算时，每个簇成员通过专用端口定期向簇头发送在线公告，以显示无人机的连接状态。该在线公告消息的数据量非常小，本系统将发送周期为2秒，以减少通信开销。此外，本系统还设置了一个计时器来判定无人机的连接是否断开。当没有收到公告消息时，计时器的计数会减少，并在收到在线公告消息后重置数值。值得注意的是，本系统不仅可以监测簇内成员的状态，簇头的状态还可以通过其他簇头进行监测。
<ul><li><b>机间通信速率测量</b></li></ul>
<p>由于无人机群中的无人机通常需要同时与多架无人机通信，无人机之间的通信速率通常小于无人机的总通信速率。为此，本系统通过记录每次通信的时间开销，并计算数据量与通信时间的比值作为当前无人机间的通信速率。具体而言，无人机间通信速率R由以下公式更新，该公式使用系数α平滑极端数据，该系数在本系统中设置为0.2。
</p>
<div align="center">
    <img src="./img/formula 1.png" height="25px"></img>
</div>
<br>
</details>
<h2 id="requirements">📦️ 软件需求</h2>

ASAP是用python编写的，目前在Linux系统上运行。运行程序之前应安装`python3`。

要顺利地使用ASAP，需要安装相应的必需软件包。经过测试的软件包版本要求列在下表中：

| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 软件包 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 版本要求 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| :----------: | :-----------:|
| torch | `torch==1.10.0`, `torchvision==0.11.1` |
| numpy | `numpy<=1.19.0` |
| yaml | `PyYAML==6.0.1` |
| scipy | `scipy==0.19.1` |
| scikit-learn | `scikit-learn==0.24.2` |

<h2 id="usage">🚀 如何使用</h2>

<h3>从github.com上克隆代码</h3>

```
git clone https://github.com/snhao222/ASAP.git
cd ASAP
```

<h3>为你的设备构建深度学习推理性能预测器</h3>

生成算子级预测参数。

```
cd Predictor_tool

# 测量卷积类算子的阶梯形状
# 这会生成一个.csv文件'conv_pattern_measurement.csv'
python3 conv_pattern_measurement.py

# 为算子级预测器测量卷积参数
# 在开始前请根据上一步测得的图形修改阶梯的步长值（step length）
# 这会生成一个配置文件'para.yml'
python3 conv_measurement.py

# 测量线拟合参数
# 这会生成一个.csv文件'other_operators.csv'
python3 ratio_fitting.py
# 生成配置文件'fitting.yml'
python3 data_process.py
```

为时延融合微调器生成训练集和验证集。

```
# 你可以通过运行以下文件检测算子融合规则
# 这会生成一个.yml文件'fusion_rules.yml'
python3 fusion_rule_detection.py

# 测量算子对的推理时延数据集
# 这会生成一个.csv文件'local_latency.csv'
python3 model_construct.py
# 一个进度条会实时显示测量进度
```

<div align="center">
    <img src="./img/dataset generator.gif" width="800px"></img>
</div>


训练时延融合微调模型。

```
# 运行以下文件并生成一个.pt文件'fusion_predictor.pt'
python3 fusion_predictor.py
# 训练集和验证集的损失值将会被实时显示
```

<div align="center">
    <img src="./img/trainning graph.gif" width="700px"></img>
</div>


为方便快速试用，本项目提供了实验中用到的预训练预测器模型参数和任务模型（即 VGG16, ResNet34, ResNet50, DarkNet19）。你可以点击 [这里](https://drive.google.com/file/d/1dDlgt9giXh9_7gbtBN06ZgUqVYZCNBIP/view?usp=sharing) 进行下载.



<h3>在单个节点上开始部署</h3>

```
cd ..
# 使用'-h'查看帮助信息
python3 master.py -h
# -c , --cluster   Cluster number
# -r , --role      'H' for cluster head, and 'M' for cluster member
# -t , --task      'vgg' for VGG16; 'resnet34' for ResNet34; 'resnet50' for
#                  ResNet50; 'resnet101' for ResNet101; 'resnet152' for
#                  ResNet152; 'darknet' for DarkNet19
# -o, --owner      Default 'False'
# -e, --elastic    Default 'False'

# 示例:
# 	对于簇1中的簇头（任务发布节点）
# 	计算模型为ResNet152，你可以运行以下命令
# 	>>> python3 master.py -c 1 -r H -t resnet152 -o
```

<h3>多节点协同</h3>

在所有节点都启动后，节点将自动完成相互接入。

在接入过程结束后，在任务发布的节点上点击“Enter”键以启动协同计算程序。需要注意的是，其余节点在此之后将自动启动协同计算程序，无需其他操作。

<h2 id="contributing">📫 帮助和建议</h2>

本项目非常欢迎相关领域研究人员的贡献和建议。

<h2 id="license">📑 许可证</h2>

本项目源码在MIT许可证范围内开源。

<h2 id="citation">✏️ 引用本工作</h2>

如果ASAP对你的研究工作有帮助，请考虑引用以下论文：

```
@inproceedings{asap,
  title={All Sky Autonomous Computing in UAV Swarm},
# 完整引用BibTeX在出版后更新。
}
```