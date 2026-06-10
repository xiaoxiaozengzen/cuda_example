# cuda_example

使用cuda、cudnn、tensorrt的例子

# cuda
cuda(Compute Unified Device Architecture):NVIDIA的统一的设备计算架构，提供接口，用于访问和使用GPU。CUDA只能在NVIDIA的GPU上运行

# cudnn
cudnn(CUDA Deep Neural Network)：基于CUDA的深度神经网络加速库，提供卷积、归一化等高性能实现。是NVIDIA打造的针对深度神经网络的加速库，是一个用于深层神经网络的GPU加速库。它能将模型训练的计算优化之后，再通过 CUDA 调用 GPU 进行运算.
Cudnn通常和CUDA是版本绑定的。

# TensorRT
TensorRT：英伟达针对自家平台做的加速包，只负责模型的推理（inference）过程，一般不用TensorRT来训练模型的，而是用于部署时加速模型运行速度。

TensorRT主要做了这么两件事情，来提升模型的运行速度。
* TensorRT支持INT8和FP16的计算。深度学习网络在训练时，通常使用 32 位或 16 位数据。TensorRT则在网络的推理时选用不这么高的精度，达到加速推断的目的。
* TensorRT对于网络结构进行了重构，把一些能够合并的运算合并在了一起，针对GPU的特性做了优化。现在大多数深度学习框架是没有针对GPU做过性能优化的，而英伟达，GPU的生产者和搬运工，自然就推出了针对自己GPU的加速工具TensorRT。一个深度学习模型，在没有优化的情况下，比如一个卷积层、一个偏置层和一个reload层，这三层是需要调用三次cuDNN对应的API，但实际上这三层的实现完全是可以合并到一起的，TensorRT会对一些可以合并网络进行合并。

# nsight

nsys：Nsight Systems 是 NVIDIA 提供的一款性能分析工具，主要用于分析 GPU 和 CPU 之间的工作负载，以帮助开发者找到瓶颈并优化性能。它可以提供从 CPU 调度到 GPU 内核执行的详细时序视图，帮助识别性能瓶颈。

## 下载

```bash
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_6/nsight-systems-2025.6.1_2025.6.1.190-1_amd64.deb
```

## 使用

### 基本使用

```bash
# 这个命令会生成一个 .qdrep或者.nsys-rep 格式的报告文件，可以使用GUI工具进行分析
nsys profile --stats=true -o report_name ./your_program
# profile 是 Nsight Systems 的主要命令，表示进行性能分析。
# --stats=true 表示在分析后打印统计信息。
# -o report_name 表示生成的报告文件名。
# ./your_program 是要分析的可执行程序。
```

### 可视化

```bash
nsys-ui <your_report_path>.nsys-rep
```