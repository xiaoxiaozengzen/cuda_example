# Overview

`cudaIpcMemhandle_t`是cuda的IPC机制，它将一块GPU内存保存成句柄，然后其他进程获取这个句柄后再转成GPU地址使用

大致的流程：
1. server端调用cuadmalloc分配内存
2. server调用cudaIpcGetMemHandle把这块内存导出成一个handle
3. server把这个handle通过socket/pipe等进程共享手段分享给client
4. client收到handle后调用cudaIpcOpenMemHandle，得到一个本地进程可用的设备指针
5. client像使用普通的device pointer去使用
6. client调用完之后使用cudaIpcCloseMemHandle
7. server最后再cudaFree原始内存

适合的场景：
1. 两个linux进程之间共享GPU内存
2. 避免D2H，H2D拷贝
3. 视频流、图像、Tensor的零拷贝

基本限制：
1. client跟server要在同一块支持IPC的GPU上
2. server不能再client还在使用时提前cudaFree
3. client打开的映射记得cudaIpcCloseMemHandle去关闭
4. 传递的是handle，不是裸指针