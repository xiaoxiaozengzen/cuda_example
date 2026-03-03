# Overview

# 检查模型可用性

```bash
# 对onnx模型做基线检查
trtexec --onnx=deploy.onnx --verbose --minShapes=input:1x10 --optShapes=input:10x10 --maxShapes=input:100x10

# 将onnx模型转换成tensorRT模型
trtexec --onnx=/mnt/workspace/cgz_workspace/Exercise/python_example/pytorch/deploy.onnx --saveEngine=deploy.engine  --verbose --minShapes=input:1x10 --optShapes=input:10x10 --maxShapes=input:100x10

# 执行下列命令，不报错就证明模型和当前平台是适配的
trtexec --loadEngine=xxx.engine --verbose
```