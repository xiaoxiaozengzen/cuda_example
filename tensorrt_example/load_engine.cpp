#include <fstream>
#include <iostream>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

/**
 * 大致流程：
 * 1. 反序列化engine
 * 2. 创建context
 * 3. 设置动态输入的shape
 * 4. 绑定显存地址
 * 5. enqueue
 * 6. 拷贝回结果
 */

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error in " << __FILE__ << " at line "      \
                      << __LINE__ << ": " << cudaGetErrorString(err)      \
                      << " (" << err << ")" << std::endl;                 \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << msg << "\n";
    }
} gLogger;

static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if(!f) {
        std::cerr << "Failed to open file: " << path << "\n";
        return {};
    }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> data(sz);
    f.read(data.data(), sz);
    return data;
}

static size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

int main() {
    std::string engine_file = "/mnt/workspace/cgz_workspace/Exercise/python_example/pytorch/deploy.engine";
    auto engineData = readFile(engine_file);
    if(engineData.empty()) {
        std::cerr << "Failed to read engine file\n";
        return -1;
    }
    std::cout << "Engine file read successfully, size: " << engineData.size() << " bytes\n";

    /**
     * @brief Create TensorRT runtime
     * IRuntime是TensorRT的核心接口，用于管理引擎的生命周期和执行环境。通过反序列化引擎数据创建ICudaEngine实例。
     * - createInferRuntime：创建一个TensorRT运行时对象，负责管理引擎的生命周期和执行环境。
     * - IRuntime::deserializeCudaEngine：将反序列化的引擎数据转换为ICudaEngine实例，准备执行推理。
     * - IRuntime::createExecutionContext：为引擎创建一个执行上下文，负责管理推理过程中输入输出的绑定和执行状态
     */
    auto runtime = nvinfer1::createInferRuntime(gLogger);
    if(!runtime) {
        std::cerr << "Failed to create TensorRT runtime\n";
        return -1;
    }

    /**
     * @brief 反序列化一个engine
     * @param blob 包含序列化引擎数据的内存块
     * @param size 内存块的大小
     * @param pluginFactory 可选的插件工厂，用于处理引擎中使用的自定义层，新版本中已废弃
     * @return ICudaEngine* 反序列化后的ICudaEngine实例，如果失败则返回nullptr
     *
     * @note ICudaEngine用于执行推理，包含了网络结构、权重和执行配置等信息。
     */
    auto engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if(!engine) {
        std::cerr << "Failed to deserialize CUDA engine\n";
        return -1;
    }

    /**
     * @brief 为引擎创建一个执行上下文
     * @return IExecutionContext* 创建的执行上下文，如果失败则返回nullptr
     *
     * @note IExecutionContext用于管理推理过程中输入输出的绑定和执行状态。
     * @note 如果engine是动态的（包含动态输入），则需要在执行上下文中设置profile。
     *       第一个创建的IExecutionContext会默认调用setOptimizationProfile(0)，如果有多个profile需要切换，必须显式
     *       调用setOptimizationProfile()方法来选择适当的profile，以确保输入输出的shape在执行时得到正确处理。
     */
    auto context = engine->createExecutionContext();
    if(!context) {
        std::cerr << "Failed to create execution context\n";
        return -1;
    }

    const char* inputName = nullptr;
    const char* outputName = nullptr;

    /**
     * @brief 获取引擎的输入输出信息
      - getNbIOTensors：返回引擎中总的输入和输出张量的数量。
      - getIOTensorName：根据索引获取输入输出张量的名称。
      - getTensorIOMode：根据张量名称，判断其是输入还是输出张量，返回TensorIOMode枚举值（kINPUT或kOUTPUT或kNONE）。
     */
    std::cout << "Engine has " << engine->getNbIOTensors() << " I/O tensors\n";
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        std::cout << "Tensor " << i << ": " << name << "\n";
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputName = name;
        } else {
            outputName = name;
        } 
    }

    int N = 10;  // batch

    /**
     * @brief 用于定义tensor的维度
     * class Dims32 {
     * public:
     *     static constexpr int32_t MAX_DIMS{8}; // 一个张量的最大维度数量
     *     int32_t nbDims; // 实际维度数量
     *     int32_t d[MAX_DIMS]; // 每个维度的大小
     * }
     * using Dims = Dims32; // TensorRT中常用的维度类型
     */
    nvinfer1::Dims inDims;
    inDims.nbDims = 2;
    inDims.d[0] = N;
    inDims.d[1] = 10;

    /**
     * @brief 设置输入张量的动态shape
     * @param tensorName 输入张量的名称，必须是null-terminated字符串
     * @param shape 输入张量的维度信息
     * @return bool 成功返回true，失败返回false
     */
    if (!context->setInputShape(inputName, inDims)) {
        std::cerr << "setInputShape failed\n";
        return -1;
    }

    /**
     * @brief 获取给定的输入或者输出张量的维度信息
     * @param tensorName 输入或者输出张量的名称，必须是null-terminated字符串
     * @return Dims 获取到的维度信息，如果张量名称无效或者维度信息未解析，则返回一个Dims{-1, {}}的对象
     */
    auto outDims = context->getTensorShape(outputName); // 应该是 [N,1]
    std::cout << "Output shape: [";
    for (int i = 0; i < outDims.nbDims; ++i) {
        std::cout << outDims.d[i] << (i < outDims.nbDims - 1 ? ", " : "");
    }
    std::cout << "]\n";
    if (outDims.d[0] < 0) {
        std::cerr << "Output shape unresolved\n";
        return -1;
    }

    size_t inCount = static_cast<size_t>(N) * 10;
    size_t outCount = volume(outDims);

    std::vector<float> hIn(inCount, 0.5f);
    std::vector<float> hOut(outCount, 0.0f);

    float* dIn = nullptr;
    float* dOut = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dIn), inCount * sizeof(float)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dOut), outCount * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUDA(cudaMemcpyAsync(dIn, hIn.data(), inCount * sizeof(float), cudaMemcpyHostToDevice, stream));

    /**
     * @brief 设置输入输出张量的显存地址
     * @param tensorName 输入或者输出张量的名称，必须是null-terminated字符串
     * @param address 输入或者输出张量在GPU上的显存地址
     * @return bool 成功返回true，失败返回false
     *
     * @note 传入nullptr来reset之前设置的地址。如果input point是void const*，则使用setInputTensorAddress
     */
    context->setTensorAddress(inputName, dIn);
    context->setTensorAddress(outputName, dOut);

    /**
     * @brief 异步执行推理
     * @param stream CUDA流，用于异步执行推理kernels
     * @return bool 成功返回true，失败返回false
     */
    if (!context->enqueueV3(stream)) {
        std::cerr << "enqueueV3 failed\n";
        return -1;
    }

    CHECK_CUDA(cudaMemcpyAsync(hOut.data(), dOut, outCount * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::cout << "Output:\n";
    for (size_t i = 0; i < outCount; ++i) {
        std::cout << hOut[i] << " ";
    }
    std::cout << "\n";

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaStreamDestroy(stream));
    delete context;
    delete engine;
    delete runtime;
    return 0;
}