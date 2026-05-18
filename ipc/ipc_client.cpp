#include <cuda_runtime.h>

#include <cerrno>
#include <cstring>
#include <iostream>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                      \
    } while (0)

namespace {
constexpr char kSocketPath[] = "/tmp/cuda_ipc_demo.sock";
constexpr int kCount = 16;
}

int main() {
    // 1. 创建 Unix 域套接字
    int client_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (client_fd < 0) {
        std::cerr << "socket failed: " << std::strerror(errno) << std::endl;
        return 1;
    }

    // 2. 连接服务器套接字
    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", kSocketPath);
    if (connect(client_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "connect failed: " << std::strerror(errno) << std::endl;
        close(client_fd);
        return 1;
    }

    // 3. 从服务器接收 IPC 内存句柄
    cudaIpcMemHandle_t handle {};
    ssize_t read_bytes = read(client_fd, &handle, sizeof(handle));
    if (read_bytes != static_cast<ssize_t>(sizeof(handle))) {
        std::cerr << "read handle failed" << std::endl;
        close(client_fd);
        return 1;
    }

    // 4. 使用 IPC 句柄打开远程设备内存
    int* remote_dev_ptr = nullptr;
    CHECK_CUDA(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&remote_dev_ptr),
                                    handle,
                                    cudaIpcMemLazyEnablePeerAccess));

    // 5. 从远程设备内存复制数据到主机并打印
    int host_data[kCount] = {};
    CHECK_CUDA(cudaMemcpy(host_data,
                          remote_dev_ptr,
                          sizeof(host_data),
                          cudaMemcpyDeviceToHost));

    std::cout << "client: received GPU data:" << std::endl;
    for (int i = 0; i < kCount; ++i) {
        std::cout << host_data[i] << (i + 1 == kCount ? '\n' : ' ');
    }

    // 6. 关闭 IPC 内存句柄并清理资源
    CHECK_CUDA(cudaIpcCloseMemHandle(remote_dev_ptr));
    close(client_fd);
    return 0;
}