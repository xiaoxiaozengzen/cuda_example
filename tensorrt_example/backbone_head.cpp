
#include <bits/stdc++.h>

#include <bits/c++config.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <unordered_map>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

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

float SoftMax(float input) {
    return 1.0f / (1.0f + std::exp(-input));
}

std::pair<int, double> GetOutput(std::vector<float> output) {
    // 1. softmax
    std::vector<float> ret(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        ret[i] = SoftMax(output[i]);
    }
    // 2. 找到最大值的索引和对应的概率
    auto max_it = std::max_element(ret.begin(), ret.end());
    int max_index = std::distance(ret.begin(), max_it);
    double max_prob = *max_it;
    return std::make_pair(max_index, max_prob);
}

void with_graph() {
    // 1. 读取backbone和head的TensorRT模型文件
    std::string backbone_file = "/mnt/workspace/cgz_workspace/Exercise/cuda_example/tensorrt_example/model/backbone.trt";
    std::string head_file = "/mnt/workspace/cgz_workspace/Exercise/cuda_example/tensorrt_example/model/head.trt";
    std::vector<char> backbone_data = readFile(backbone_file);
    if(backbone_data.empty()) {
        std::cerr << "Failed to parse file: " << backbone_file << std::endl;
        return;
    }
    std::cout << "Backbone engine file read successfully, size: " << backbone_data.size() << " bytes\n";
    std::vector<char> head_data = readFile(head_file);
    if(head_data.empty()) {
        std::cerr << "Failed to parse file: " << head_file << std::endl;
        return;
    }
    std::cout << "Head engine file read successfully, size: " << head_data.size() << " bytes\n";

    // 2. 创建TensorRT运行时和引擎
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if(!runtime) {
        std::cerr << "Failed to create TensorRT runtime\n";
        return;
    }
    std::cout << "TensorRT runtime created successfully\n";
    nvinfer1::ICudaEngine* backbone_engine = runtime->deserializeCudaEngine(backbone_data.data(), backbone_data.size());
    if(!backbone_engine) {
        std::cerr << "Failed to deserialize CUDA engine\n";
        return;
    }
    std::cout << "Backbone engine deserialized successfully\n";
    nvinfer1::IExecutionContext* backbone_context = backbone_engine->createExecutionContext();
    if(!backbone_context) {
        std::cerr << "Failed to create execution context\n";
        return;
    }
    std::cout << "Backbone execution context created successfully\n";
    nvinfer1::ICudaEngine* head_engine = runtime->deserializeCudaEngine(head_data.data(), head_data.size());
    if(!head_engine) {
        std::cerr << "Failed to deserialize CUDA engine\n";
        return;
    }
    std::cout << "Head engine deserialized successfully\n";
    nvinfer1::IExecutionContext* head_context = head_engine->createExecutionContext();
    if(!head_context) {
        std::cerr << "Failed to create execution context\n";
        return;
    }
    std::cout << "Head execution context created successfully\n";

    // 3. 获取输入输出张量的名称和形状
    const char* backbone_inputName = nullptr;
    const char* backbone_outputName = nullptr;
    const char* head_inputName = nullptr;
    const char* head_outputName = nullptr;
    std::cout << "Backbone engine has " << backbone_engine->getNbIOTensors() << " I/O tensors\n";
    for (int i = 0; i < backbone_engine->getNbIOTensors(); ++i) {
        const char* name = backbone_engine->getIOTensorName(i);
        nvinfer1::Dims shape = backbone_engine->getTensorShape(name);
        if (backbone_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            backbone_inputName = name;
        } else {
            backbone_outputName = name;
        }
    }
    std::cout << "Head engine has " << head_engine->getNbIOTensors() << " I/O tensors\n";
    for (int i = 0; i < head_engine->getNbIOTensors(); ++i) {
        const char* name = head_engine->getIOTensorName(i);
        nvinfer1::Dims shape = head_engine->getTensorShape(name);
        if (head_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            head_inputName = name;
        } else {
            head_outputName = name;
        }
    }

    // 4. 设置维度
    int batchSize = 1;
    nvinfer1::Dims backbone_inDims = backbone_context->getTensorShape(backbone_inputName);
    std::cout << "Backbone input shape: [";
    for(int i = 0; i < backbone_inDims.nbDims; i++) {
        std::cout << backbone_inDims.d[i] << (i < backbone_inDims.nbDims - 1 ? ", " : "");
    }
    std::cout << "]\n";
    auto backbone_outDims = backbone_context->getTensorShape(backbone_outputName); // 应该是 [N, 128]
    if(backbone_outDims.d[0] <= 0) {
        std::cerr << "Backbone output shape unresolved\n";
        return;
    }
    std::cout << "Backbone output shape: [";
    for (int i = 0; i < backbone_outDims.nbDims; ++i) {
        std::cout << backbone_outDims.d[i] << (i < backbone_outDims.nbDims - 1 ? ", " : "");
    }
    std::cout << "]\n";
    nvinfer1::Dims head_inDims;
    head_inDims.nbDims = backbone_outDims.nbDims;
    for (int i = 0; i < backbone_outDims.nbDims; ++i) {
        head_inDims.d[i] = backbone_outDims.d[i];
    }
    if(!head_context->setInputShape(head_inputName, head_inDims)) {
        std::cerr << "setInputShape failed\n";
        return;
    }
    std::cout << "Head input shape: [";
    for(int i = 0; i < head_inDims.nbDims; i++) {
        std::cout << head_inDims.d[i] << (i < head_inDims.nbDims - 1 ? ", " : "");
    }
    std::cout << "]\n";
    auto head_outDims = head_context->getTensorShape(head_outputName); // 应该是 [N, 10]
    if(head_outDims.d[0] <= 0) {
        std::cerr << "Head output shape unresolved\n";
        return;
    }
    std::cout << "Head output shape: [";
    for (int i = 0; i < head_outDims.nbDims; ++i) {
        std::cout << head_outDims.d[i] << (i < head_outDims.nbDims - 1 ? ", " : "");
    }
    std::cout << "]\n";

    // 5.设置输出/输出
    float* d_backbone_in = nullptr;
    float* d_backbone_out = nullptr;
    float* d_head_in = nullptr;
    float* d_head_out = nullptr;
    size_t backbone_inCount = 1;
    for(int i = 0; i < backbone_inDims.nbDims; ++i) {
        backbone_inCount *= static_cast<size_t>(backbone_inDims.d[i]);
    }
    size_t backbone_outCount = 1;
    for(int i = 0; i < backbone_outDims.nbDims; ++i) {
        backbone_outCount *= static_cast<size_t>(backbone_outDims.d[i]);
    }
    size_t head_outCount = 1;
    for(int i = 0; i < head_outDims.nbDims; ++i) {
        head_outCount *= static_cast<size_t>(head_outDims.d[i]);
    }
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_backbone_in), backbone_inCount * sizeof(float)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_backbone_out), backbone_outCount * sizeof(float)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_head_out), head_outCount * sizeof(float)));
    backbone_context->setTensorAddress(backbone_inputName, d_backbone_in);
    backbone_context->setTensorAddress(backbone_outputName, d_backbone_out);
    head_context->setTensorAddress(head_inputName, d_backbone_out);
    head_context->setTensorAddress(head_outputName, d_head_out);

    // 6. 创建graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    if (!backbone_context->enqueueV3(stream)) {
        std::cerr << "backbone enqueueV3 failed\n";
        return;
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if(!head_context->enqueueV3(stream)){
        std::cerr << "head enqueueV3 failed\n";
        return;
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    backbone_context->enqueueV3(stream);
    head_context->enqueueV3(stream);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));

    // 7. 循环推理
    std::string image_path_pre = "/mnt/workspace/cgz_workspace/Exercise/cuda_example/tensorrt_example/input/";
    std::unordered_map<int, std::vector<float>> image_map = {
        {0, std::vector<float>(0)}, // Placeholder for digit 0
        {1, std::vector<float>(0)}, // Placeholder for digit 1
        {2, std::vector<float>(0)}, // Placeholder for digit 2
        {3, std::vector<float>(0)}, // Placeholder for digit 3
        {4, std::vector<float>(0)}, // Placeholder for digit 4
        {5, std::vector<float>(0)}, // Placeholder for digit 5
        {6, std::vector<float>(0)}, // Placeholder for digit 6
        {7, std::vector<float>(0)}, // Placeholder for digit 7
        {8, std::vector<float>(0)}, // Placeholder for digit 8
        {9, std::vector<float>(0)}  // Placeholder for digit 9
    };
    for(int i = 0; i < 10; ++i) {
        std::string image_path = image_path_pre + std::to_string(i) + ".bin";
        std::vector<char> image_data_char = readFile(image_path);
        if(image_data_char.empty()) {
            std::cerr << "Failed to read image file: " << image_path << "\n";
            continue;
        }
        if(image_data_char.size() != 28 * 28 * sizeof(float)) {
            std::cerr << "Unexpected image size for file: " << image_path << "\n";
            continue;
        }
        std::vector<float> image_data_float(28 * 28);
        memcpy(image_data_float.data(), image_data_char.data(), 28 *28 * sizeof(float));
        image_map[i] = image_data_float;
    }

    std::vector<float> res(head_outCount, 0.0f);
    for (int i = 0; i < 10; ++i) {
        if(image_map[i].empty()) {
            std::cerr << "Image data for digit " << i << " is empty, skipping\n";
            continue;
        }
        auto& image_data = image_map[i];
        CHECK_CUDA(cudaMemcpyAsync(d_backbone_in, image_data.data(), backbone_inCount * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaMemcpyAsync(res.data(), d_head_out, head_outCount * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        auto ret = GetOutput(res);
        std::cout << "ActualClass: " << i << ", PredictedClass: " << ret.first << ", Prob: " << ret.second << "\n";
    }

    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaFree(d_backbone_in));
    CHECK_CUDA(cudaFree(d_backbone_out));
    CHECK_CUDA(cudaFree(d_head_out));
    CHECK_CUDA(cudaStreamDestroy(stream));
    delete backbone_context;
    delete backbone_engine;
    delete head_context;
    delete head_engine;
    delete runtime;    
}

int main(int argc, char** argv) {
    std::cout << "========================= with_graph =========================\n";
    with_graph();

    return 0;
}