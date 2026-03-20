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
        if(d.d[i] < 0) {
            std::cerr << "Invalid dimension size: " << d.d[i] << "\n";
            throw std::runtime_error("Invalid dimension size");
            return 0;
        }
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

/**
 * @brief 计算给定张量格式和元素大小的内存需求
 * @param  dims             张量的维度信息
 * @param  format           张量的存储格式
 * @param  elementSize      每个元素的大小（字节）
 * @return size_t           所需的内存大小（字节）
 *
 * @note tensorFormat会影响TensorRT在内存中如何存储这个tensor，
 *       例如是否需要对齐，提高kernel执行效率等。
 */
size_t adaptTensorFormatAlloc(const nvinfer1::Dims& dims, nvinfer1::TensorFormat format, size_t elementSize) {
    size_t vol = volume(dims);
    switch (format) {
        // 线性格式通常不需要额外的对齐
        case nvinfer1::TensorFormat::kLINEAR:
        // HWC格式一般不需要额外的对齐
        case nvinfer1::TensorFormat::kHWC:
            return vol * elementSize;
        case nvinfer1::TensorFormat::kCHW2:
        {
            size_t old_c = dims.d[1]; // C
            size_t new_c = ((old_c + 2 - 1) / 2) * 2; // 向上对齐到2的倍数
            size_t new_volume = dims.d[0] * new_c * dims.d[2] * dims.d[3]; // N * C' * H * W
            return new_volume * elementSize;
        }
        case nvinfer1::TensorFormat::kHWC8:
        {
            size_t old_c = dims.d[1]; // C
            size_t new_c = ((old_c + 8 - 1) / 8) * 8; // 向上对齐到8的倍数
            size_t new_volume = dims.d[0] * dims.d[2] * dims.d[3] * new_c; // N * H * W * C'
            return new_volume * elementSize;
        }
        default:
            return vol * elementSize; // 默认情况下按线性格式计算
    }

}

int main() {
    std::string engine_file = "/mnt/workspace/cgz_workspace/Exercise/cuda_example/tensorrt_example/input/deploy.engine";
    std::vector<char> engineData = readFile(engine_file);
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
     *
     * @note createInferRuntime可以多次调用，每次创建一个实例。但是推荐一个进程只调用一次
     */
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
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
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
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
     *       第一个创建的IExecutionContext会默认调用setOptimizationProfile(0)
     * @note 动态shape的engine一般有多个profile(例如，小batch/大batch)
     *       如果有多个profile需要切换，必须显式调用setOptimizationProfile()方法来选择适当的profile，
     *       以确保输入输出的shape在执行时得到正确处理。并且profile也会影响trt选用的内核和性能策略等
     */
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
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
    int profile_num = engine->getNbOptimizationProfiles();
    std::cout << "Engine has " << profile_num << " optimization profiles\n";
    std::cout << "Engine has " << engine->getNbIOTensors() << " I/O tensors\n";
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        std::cout << "Tensor " << i << ": " << name << "\n";
        nvinfer1::Dims shape = engine->getTensorShape(name);
        std::cout << "  Shape(" << shape.nbDims << "): [";
        for (int j = 0; j < shape.nbDims; ++j) {
            std::cout << shape.d[j] << (j < shape.nbDims - 1 ? ", " : "");
        }
        std::cout << "]\n";
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputName = name;
            /**
             * 获取一个input tensor的min/opt/max维度信息基于其使用的profile
             */
            nvinfer1::Dims profile_shape = engine->getProfileShape(name, profile_num - 1, nvinfer1::OptProfileSelector::kMAX); // profile 0
            std::cout << "  Profile shape: [";
            for (int j = 0; j < profile_shape.nbDims; ++j) {
                std::cout << profile_shape.d[j] << (j < profile_shape.nbDims - 1 ? ", " : "");
            }
            std::cout << "]\n";
        } else {
            outputName = name;
        }
        /**
         * enum class DataType : int32_t {
         *   kFLOAT = 0,   // 32-bit floating point
         *   kHALF = 1,    // 16-bit floating point
         *   kINT8 = 2,    // 8-bit integer
         *   kINT32 = 3,   // 32-bit integer
         *   kBOOL = 4     // Boolean
         * }
         */
        nvinfer1::DataType dtype = engine->getTensorDataType(name);
        std::cout << "  Data type: " << static_cast<int>(dtype) << "\n";
        /**
         * enum class TensorFormat : int32_t {
         *   kLINEAR = 0,  // 行优先的线性格式，适用于大多数情况
         *   kCHW2 = 1,    // CHW2
         *   kHWC8 = 2,    // HWC8
         *   kCHW4 = 3,    // CHW4
         *   kHWC16 = 4,   // HWC16
         *   kDLA_HWC4 = 5 // DLA HWC4
         * }
         * @note NCHW，表示[numbers, channels, height, width]，其w元素是连续的，即：
         *       w+1，地址加1；h+1，地址加w；c+1，地址加w*h；n+1，地址加w*h*c
         * @note HWC，C数据是连续的，即x[h][w][0]=R, x[h][w][1]=G, x[h][w][2]=B
         * @note CHW，x[0,:,:]R通道整张图，x[1,:,:]G通道整张图，x[2,:,:]B通道整张图
         * @note HWC8，C的数量按照8对齐，例如C=3时会补齐到8，实际存储元素数为 H*W*8，访问时需要跳过补齐的元素
         * @note CHW4，C的数量按照4对齐，例如C=3时会补齐到4，实际存储元素数为 4*H*W，访问时需要跳过补齐的元素
         */
        nvinfer1::TensorFormat format = engine->getTensorFormat(name);
        std::cout << "  Tensor format: " << static_cast<int>(format) << "\n";
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