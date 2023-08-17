#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter_builder.h>

struct ImageClassifier {
    ImageClassifier(std::string_view model_path, std::string_view labels_path);

    [[nodiscard]] std::vector<std::pair<double, std::string>> run(const cv::Mat& image) const noexcept;

private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

    std::vector<std::string> labels;

    int image_height{0};
    int image_width{0};
};

