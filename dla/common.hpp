// Shared utilities for DLA example: TensorRT logger, CUDA error checking, file I/O.
// Callers reference every symbol with a fully qualified name (dla_example::X, std::X, nvinfer1::X)
// so it's always obvious which library a call comes from.

#pragma once

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace dla_example {

// Abort on CUDA error. Simple; a demo doesn't need exceptions.
#define DLA_EXAMPLE_CUDA_CHECK(expr)                                                              \
    do {                                                                                          \
        cudaError_t status = (expr);                                                              \
        if (status != cudaSuccess) {                                                              \
            std::cerr << "CUDA error " << cudaGetErrorString(status) << " at " << __FILE__ << ":" \
                      << __LINE__ << " (" #expr ")" << std::endl;                                 \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

// TensorRT logger. Prints WARNING and above to stderr.
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            const char* level = "INFO";
            switch (severity) {
                case Severity::kINTERNAL_ERROR: level = "INTERNAL_ERROR"; break;
                case Severity::kERROR:          level = "ERROR"; break;
                case Severity::kWARNING:        level = "WARNING"; break;
                case Severity::kINFO:           level = "INFO"; break;
                case Severity::kVERBOSE:        level = "VERBOSE"; break;
            }
            std::cerr << "[TRT " << level << "] " << msg << std::endl;
        }
    }
};

inline std::vector<char> ReadBinaryFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "Failed to open file for reading: " << path << std::endl;
        std::exit(1);
    }
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    if (!f.read(buf.data(), size)) {
        std::cerr << "Failed to read file: " << path << std::endl;
        std::exit(1);
    }
    return buf;
}

inline void WriteBinaryFile(const std::string& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
        std::exit(1);
    }
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
}

inline int64_t VolumeOf(const nvinfer1::Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        v *= dims.d[i];
    }
    return v;
}

inline size_t ElementSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        case nvinfer1::DataType::kUINT8: return 1;
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kFP8:   return 1;
        case nvinfer1::DataType::kBF16:  return 2;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kINT4:  return 1;
#endif
    }
    return 0;
}

inline std::string DimsToString(const nvinfer1::Dims& dims) {
    std::string s = "(";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) s += ",";
        s += std::to_string(dims.d[i]);
    }
    s += ")";
    return s;
}

}  // namespace dla_example
