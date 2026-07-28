# Overview

DLA(DeepLearningAccelerator)，深度学习加速器

是一个独立的硬件，再orin/thor上。

区别GPU：

* 不接受动态shape
* 只支持int8后者fp16。 其中使用int8需要QAT(Quantization-Aware Training)，让模型知道自己要被训练成INT8跑
* 支持的算子有限，不支持自定义的算子
