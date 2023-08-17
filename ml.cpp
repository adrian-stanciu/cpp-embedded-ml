#include "ml.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/core.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>

ImageClassifier::ImageClassifier(std::string_view model_path, std::string_view labels_path)
{
    // load model
    [this, &model_path]() {
        model = tflite::FlatBufferModel::BuildFromFile(model_path.data());
        if (!model)
            throw std::runtime_error("failed to load model");
    }();

    // build interpreter
    [this]() {
        static constexpr auto NumThreads{2};

        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);

        if (builder.SetNumThreads(NumThreads) != kTfLiteOk)
            fmt::print(stderr, "failed to set the number of threads to {:d}\n", NumThreads);

        builder(&interpreter);
        if (!interpreter)
            throw std::runtime_error("failed to build interpreter");

        // allocate tensors
        if (interpreter->AllocateTensors() != kTfLiteOk)
            throw std::runtime_error("failed to allocate tensors");
    }();

    // load labels
    [this, &labels_path]() {
        std::ifstream labels_ifs{labels_path.data()};
        if (!labels_ifs.is_open())
            throw std::runtime_error("failed to load labels");

        std::string label;
        while (getline(labels_ifs, label))
            if (!label.empty())
                labels.push_back(std::move(label));
    }();

    // retrieve required image dimensions
    [this]() {
        auto inputs{interpreter->inputs()};
        if (inputs.empty())
            throw std::runtime_error("interpreter has no inputs");

        auto *input_tensor{interpreter->tensor(inputs.front())};
        if (!input_tensor)
            throw std::runtime_error("no input tensor");

        auto *input_dims{input_tensor->dims};
        if (!input_dims)
            throw std::runtime_error("no dims for input tensor");

        if (input_dims->size < 3)
            throw std::runtime_error("not enough dims for input tensor");

        image_height = input_dims->data[1];
        image_width = input_dims->data[2];
    }();

    // validate output size
    [this]() {
        auto outputs{interpreter->outputs()};
        if (outputs.empty())
            throw std::runtime_error("interpreter has no outputs");

        auto *output_tensor{interpreter->tensor(outputs.front())};
        if (!output_tensor)
            throw std::runtime_error("no output tensor");

        auto *output_dims{output_tensor->dims};
        if (!output_dims)
            throw std::runtime_error("no dims for output tensor");

        if (output_dims->size < 1)
            throw std::runtime_error("not enough dims for output tensor");

        auto output_size{static_cast<size_t>(output_dims->data[output_dims->size - 1])};
        if (output_size != labels.size())
            throw std::runtime_error(fmt::format("mismatch between output size ({:d}) and number of labels ({:d})",
                output_size, labels.size()));
    }();
}

namespace {
    template <typename T>
    void copy_image_to_tensor_data(const cv::Mat& image, T *tensor_data)
    {
        if constexpr (std::is_same_v<T, uint8_t>) {
            std::memcpy(tensor_data, image.data, image.total() * image.elemSize());
        } else if constexpr (std::is_same_v<T, float>) {
            auto idx = 0;
            for (auto row = 0; row < image.rows; ++row)
                for (auto col = 0; col < image.cols; ++col) {
                    auto pixel = image.at<cv::Vec3b>(row, col);
                    for (auto ch = 0; ch < image.channels(); ++ch)
                        tensor_data[idx++] = pixel.val[ch];
                }
        }
    }

    template <typename T>
    [[nodiscard]] auto get_results(std::span<T> confidences, const std::vector<std::string>& labels)
    {
        static constexpr auto ConfidenceThreshold{0.1};

        // results are expressed as (confidence, label) pairs, where confidence is between 0.0 and 1.0
        std::vector<std::pair<double, std::string>> results;

        for (size_t label_idx{0}; label_idx < confidences.size(); ++label_idx) {
            auto confidence{1.0 * confidences[label_idx]};
            if constexpr (std::is_same_v<T, uint8_t>) {
                confidence /= std::numeric_limits<uint8_t>::max();
            }

            if (std::isnan(confidence))
                continue;
            if (confidence < ConfidenceThreshold)
                continue;

            results.emplace_back(confidence, labels[label_idx]);
        }

        // sort results in non-increasing order of confidence
        std::sort(results.rbegin(), results.rend());

        return results;
    }
}

[[nodiscard]] std::vector<std::pair<double, std::string>> ImageClassifier::run(const cv::Mat& image) const noexcept
{
    // resize image to required dimensions
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size{image_width, image_height});

    // copy resized image to input tensor
    auto input_tensor_type{interpreter->tensor(interpreter->inputs().front())->type};
    switch (input_tensor_type) {
    case kTfLiteUInt8:
        copy_image_to_tensor_data(resized_image, interpreter->typed_input_tensor<uint8_t>(0));
        break;
    case kTfLiteFloat32:
        copy_image_to_tensor_data(resized_image, interpreter->typed_input_tensor<float>(0));
        break;
    default:
        fmt::print(stderr, "input tensor type {:d} not supported\n", input_tensor_type);
        return {};
    }

    // run inference
    auto from_ts{std::chrono::steady_clock::now()};
    if (interpreter->Invoke() != kTfLiteOk) {
        fmt::print(stderr, "failed to run inference\n");
        return {};
    }
    auto to_ts{std::chrono::steady_clock::now()};

    auto duration_ms{std::chrono::duration_cast<std::chrono::milliseconds>(to_ts - from_ts)};
    fmt::print(stdout, "inference duration: {:d} ms\n", duration_ms.count());

    auto output_tensor_type{interpreter->tensor(interpreter->outputs().front())->type};
    switch (output_tensor_type) {
    case kTfLiteUInt8:
        return get_results(std::span{interpreter->typed_output_tensor<uint8_t>(0), labels.size()}, labels);
    case kTfLiteFloat32:
        return get_results(std::span{interpreter->typed_output_tensor<float>(0), labels.size()}, labels);
    default:
        fmt::print(stderr, "output tensor type {:d} not supported\n", output_tensor_type);
        return {};
    }
}

