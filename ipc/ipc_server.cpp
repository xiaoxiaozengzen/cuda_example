#include <cuda_runtime.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>

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
    // 1. 删除旧的 socket 文件（如果存在）
    unlink(kSocketPath);

    // 2. 创建 Unix 域套接字
    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "socket failed: " << std::strerror(errno) << std::endl;
        return 1;
    }

    // 3. 绑定套接字到文件路径
    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", kSocketPath);
    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "bind failed: " << std::strerror(errno) << std::endl;
        close(server_fd);
        return 1;
    }

    // 4. 监听连接
    if (listen(server_fd, 1) < 0) {
        std::cerr << "listen failed: " << std::strerror(errno) << std::endl;
        close(server_fd);
        return 1;
    }

    // 5. 分配设备内存
    int* dev_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&dev_ptr, kCount * sizeof(int)));
    int host_data[kCount];
    for (int i = 0; i < kCount; ++i) {
        host_data[i] = 100 + i;
    }
    CHECK_CUDA(cudaMemcpy(dev_ptr, host_data, sizeof(host_data), cudaMemcpyHostToDevice));

    // 6. 获取 IPC 内存句柄
    cudaIpcMemHandle_t handle {};
    CHECK_CUDA(cudaIpcGetMemHandle(&handle, dev_ptr));

    // 7. 等待客户端连接并发送 IPC 句柄
    std::cout << "server: waiting for client..." << std::endl;
    int client_fd = accept(server_fd, nullptr, nullptr);
    if (client_fd < 0) {
        std::cerr << "accept failed: " << std::strerror(errno) << std::endl;
        CHECK_CUDA(cudaFree(dev_ptr));
        close(server_fd);
        return 1;
    }

    // 8. 发送 IPC 句柄给客户端
    ssize_t written = write(client_fd, &handle, sizeof(handle));
    if (written != static_cast<ssize_t>(sizeof(handle))) {
        std::cerr << "write handle failed" << std::endl;
        close(client_fd);
        CHECK_CUDA(cudaFree(dev_ptr));
        close(server_fd);
        return 1;
    }

    std::cout << "server: handle sent, press Enter after client is done..." << std::endl;
    std::cin.get();

    close(client_fd);
    close(server_fd);
    unlink(kSocketPath);

    CHECK_CUDA(cudaFree(dev_ptr));
    std::cout << "server: device memory freed" << std::endl;
    return 0;
}