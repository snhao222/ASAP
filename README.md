

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
  <a href="README_ZH.md">简体中文</a>
</p>



<br>

This repository contains the system source code of the IEEE Transactions on Mobile Computing (TMC) paper All Sky Autonomous Computing in UAV swarm.

ASAP (<u>A</u>ll <u>S</u>ky <u>A</u>utonomous com<u>P</u>uting) is a collaborative inference system designed for UAV swarm. The key idea of ASAP is to collectively and dynamically “borrow” the power of nearby UAVs in the swarm to process the huge sensory payload data in real-time.

<p align="center">
  <a href="#highlights">Highlights</a> •
  <a href="#technical details">Technical Details</a> •
  <a href="#requirements">Requirements</a> •
  <a href="#usage">Usage</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a> •
  <a href="#citation">Citation</a>
</p>
:star: If you like this project, give the repository a star  : ) 

<h2 id="highlights">✨ Highlights</h2>

<p><ul><li>UAV swarm native collaborative computing architecture;</li></ul>
<ul><li>Efficient elastic scheduler to perform efficient and robust collaborative computing;</li></ul>
<ul><li>Lightweight accurate deep learning (DL) inference performance predictor;</li></ul>
<ul><li>Adaptive inter-UAV inference data compressor to adaptively decrease the intermediate data size;</li></ul>
<ul><li>Autodiscover and access between UAVs.</li></ul></p>
<h4>⚡ A killer application of UAVs: earthquake search and rescue</h4>
<div style="text-align: left;">  
  <img align="right" src="./img/Application case.png" width="550px" style="float: right; margin-left: 10px;">
  <p align= "justify">Due to the possible broken base station communication link (line ➍), existing relay UAV assisted approaches either transmit the massive on-site images back to the ground station for further processing in a <i>time-consuming manner</i> (line ➋), or analyse the images onboard with a compressed DL model in <i>low accuracy</i> (line ➌). Differently, ASAP propose to collaboratively process the data during transmission (line ➊), which could transmit <i>high accuracy</i> survivor information to the ground for <i>fast</i> decision-making in a <i>communication-efficient</i> manner.
  </p>
</div>

<h2 id="technical details">📋 Technical Details</h2>

<div align="center">
    <img src="./img/System overview.png" width="780px"></img>
</div>
The goal of ASAP is to **conduct efficient and reliable collaborative computing in the UAV swarm**, which relieves the burden of air-ground communication by transmitting valuable data results instead of raw data.
The system is composed with three modules, *i.e.*, elastic efficient task scheduler, lightweight accurate DNN block performance predictor, and adaptive inter-UAV inference data compressor. First, when there is a task to be processed, *i.e.*, a DL model and a sequence of data, the elastic efficient task scheduler partitions the task to UAV clusters and UAVs inside to conduct efficient collaborative computing. Besides, the allocation scheme can be updated by the task scheduler when the status of UAVs varies, *e.g.*, some UAVs become unavailable or rejoin the swarm. Second, the task allocation scheme is developed with the help of the DL inference performance predictor, which can estimate the inference latency of various model and data partitions on different UAVs at a clip. Finally, the intermediate data transmitted between UAVs will be further compressed by the adaptive inter-UAV data compressor to save the inter-UAV communication resource.

<details> <summary><b>System Module Design</b> (Unfold for details)</summary>
 The module composition of the system is shown in the below figure. We classify modules into four categories: Control, Management, Communication, and Computation. Control modules are responsible for task scheduling and data allocation. Management modules are in charge of data organization and maintenance. Communication modules are middlewares for communication management between UAVs and information transmission between modules. Computation modules handle work related to computing. It is noted that modules do not exchange information and data directly, but through the internal communication module, which ensures orderly communications between modules by unifying communication interfaces.
 <br>
 <br>
 <div align="center">
    <img src="./img/System composition.png" width="780px"></img>
 </div>
 <br>
 The relationship between main modules is illustrated in the below figure. Each UAV in the swarm is deployed with a performance measurement module to measure the communication and computing performance of UAVs, and the runtime information is collected by the state manager of cluster heads to perform elastic task scheduling. The task scheduler of task publisher UAV controls the model partition strategy and sends it to other cluster heads, and the task scheduler of cluster heads is responsible for generating data partition strategy and send to cluster members in the cluster. The data manager controls the flow of data, and distributes data partitions to the computing module for local computing and concatenates results returned from cluster members.
 <br>
 <div align="center">
    <img src="./img/System architecture.png" width="650px"></img>
 </div>
</details>

<details> <summary><b>Implementation Tricks</b> (Unfold for details)</summary>
<ul><li><b>Padding in data partition</b></li></ul>
Some layers in DL models padding zeros around feature maps to increase the perception of boundary information. However, as illustrated in the below figure, the padding operation after data partition introduces some useless information to the feature map, which affects the perception capability of DL models. To ensure the information perception ability after data partition, we customize padding strategies for DL model layers on each UAV, which are automatically generated online. In detail, when a task is received by a UAV, the model generator in the model manager will calculate the intermediate data range for each layer with the MRT algorithm, and compare the data range with the boundaries of each layer. As long as the data range exceeds the boundary, the padding of the corresponding side will be set to zero, and this process will be repeated until the entire model is traversed iteratively.
<div align="center">
    <img src="./img/Padding in data partition.png" width="300px"></img>
</div>
<ul><li><b>UAV access technique</b></li></ul>
In the proposed system, any UAV joining the swarm can be discovered automatically by other UAVs and exchange identity information with each other. The UAV access function is based on SSDP protocol (Simple Service Discovery Protocol), which allows devices to broadcast messages on a reserved address (<i>i.e.</i>, 239.255.255.250: 1900). As shown in the below figure, two kinds of messages are used to discover joined UAVs.<br>
1) M-SEARCH message: This message contains the host address and the unique service name ('usn' in short) with cluster information and UAV number.<br>
2) NOTIFY message: This message contains the location (<i>i.e.</i>, UAV local address) and the notification type ('nt' in short) with UAV information.
<br>When a UAV joins the swarm, the M-SEARCH message will be broadcast periodically, and the related UAVs will respond to the message with a NOTIFY message in unicast mode. It is noted that the UAV discovery not only occurs in the UAV cluster but also between cluster heads of different clusters.
<div align="center">
    <img src="./img/Access technique.png" width="350px"></img>
</div>
<ul><li><b>UAV status monitoring</b></li></ul>
For elastic scheduling, detecting the change of UAV status in time is crucial. Therefore, a state manager is deployed to manage the connection status of UAVs. For each cluster member, a live announcement message is sent to the cluster head periodically with a dedicated port to show the connection status of the UAV. The alive announcement message is very small, and we set the sending period to 2 seconds to reduce communication overhead. In addition, a timer is set to judge the disconnection of UAVs. The count of the timer will be decreased when not receiving the announcement message and reset after receiving the message. It is noted that not only the status of cluster members can be detected, but the cluster head can be monitored by other cluster heads.
<ul><li><b>Inter-UAV communication rate measurement</b></li></ul>
<p> Because a UAV in the swarm typically needs to communicate with multiple UAVs simultaneously, the communication rate between UAVs is usually not equal to the total communication rate. To this end, the proposed system records the communication time during each communication, and the current inter-UAV communication rate is calculated by the ratio of data volume to communication time. In detail, the inter-UAV communication rate R is updated by the following formula, which uses a coefficient α to smooth extreme data, 0.2 in our system.
</p>
<div align="center">
    <img src="./img/formula 1.png" height="25px"></img>
</div>
<br>
</details>
<h2 id="requirements">📦️ Requirements</h2>

ASAP is written in python and currently run on Linux system. `python3` should be installed before running the program.

To use ASAP successfully, the corresponding required packages are necessary to be installed. The well tested versions are listed in the table:

| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Package &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Required versions &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| :----------: | :-----------:|
| torch | `torch==1.10.0`, `torchvision==0.11.1` |
| numpy | `numpy<=1.19.0` |
| yaml | `PyYAML==6.0.1` |
| scipy | `scipy==0.19.1` |
| scikit-learn | `scikit-learn==0.24.2` |

<h2 id="usage">🚀 Usage</h2>

<h3>Clone from github.com</h3>

```
git clone https://github.com/snhao222/ASAP.git
cd ASAP
```

<h3>Build DL inference performance predictor for your device</h3>

Generate parameters for the operator level predictor.

```
cd Predictor_tool

# Measure step length of convolution-like operators' stair patterns
# A .csv file 'conv_pattern_measurement.csv' will be generated
python3 conv_pattern_measurement.py

# Measure convolution parameters for operator level predictor
# Please change the values of step lengths according to the previous step first
# A configuration file 'para.yml' will be generated
python3 conv_measurement.py

# Measure ratio fitting parameters
# Generate a .csv file 'other_operators.csv'
python3 ratio_fitting.py
# Generate a configuration file 'fitting.yml'
python3 data_process.py
```

Generate the training set and the evaluation set of latency fusion fine-tuner.

```
# You can detect the operator fusion rules by running the following file
# A .yml file 'fusion_rules.yml' will be generated
python3 fusion_rule_detection.py

# Measure inference latency set of operator pairs
# A .csv file 'local_latency.csv' will be generated
python3 model_construct.py
# A progress bar will be used to display the measurement progress
```

<div align="center">
    <img src="./img/dataset generator.gif" width="800px"></img>
</div>


Train the latency fusion fine-tuning model.

```
# Run the following file and a .pt file 'fusion_predictor.pt' will be generated
python3 fusion_predictor.py
# The loss on training set and evaluation set will be illustrated in real time
```

<div align="center">
    <img src="./img/trainning graph.gif" width="700px"></img>
</div>


The parameters of pretrained predictor model in the evaluation as well as some task models (*i.e.*, VGG16, ResNet34, ResNet50, DarkNet19) are provided as an example for quick start.  You can download them from [Here](https://drive.google.com/file/d/1dDlgt9giXh9_7gbtBN06ZgUqVYZCNBIP/view?usp=sharing).



<h3>Getting start on a single node</h3>

```
cd ..
# Use '-h' to check the help information
python3 master.py -h
# -c , --cluster   Cluster number
# -r , --role      'H' for cluster head, and 'M' for cluster member
# -t , --task      'vgg' for VGG16; 'resnet34' for ResNet34; 'resnet50' for
#                  ResNet50; 'resnet101' for ResNet101; 'resnet152' for
#                  ResNet152; 'darknet' for DarkNet19
# -o, --owner      Default 'False'
# -e, --elastic    Default 'False'

# Example:
# 	For the cluster head of cluster 1, which is the task owner,
# 	and the computing model is ResNet152, you can run the following command
# 	>>> python3 master.py -c 1 -r H -t resnet152 -o
```

<h3>Nodes collaboration</h3>

Nodes will discover each other autonomously after getting start on all the nodes.

After the discovery procedure ends, press the 'Enter' key on the task owner to start the collaborative computing procedure. Note that other nodes will start the collaborative procedure automatically, no additional operations are required.

<h2 id="contributing">📫 Contributing</h2>

This project greatly welcomes contributions and suggestions from researchers in related fields.

<h2 id="license">📑 License</h2>

The source code is under MIT license.

<h2 id="citation">✏️ Citation</h2>

If ASAP is helpful for your research, please consider citing the paper below:

```
@article{asap,
  title={All Sky Autonomous Computing in UAV Swarm},
# The full citation BibTeX will be updated after publication.
}
```