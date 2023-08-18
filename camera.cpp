#include "camera.hpp"

#include <optional>

#include <opencv2/opencv.hpp>

[[nodiscard]] std::optional<cv::Mat> retrieve_image(const char *image_path)
{
    if (image_path)
        return cv::imread(image_path);

    Camera camera;
    if (!camera.is_open()) {
        fmt::print(stderr, "failed to open camera\n");
        return std::nullopt;
    }

    while (true) {
        auto image{camera.read_image()};
        cv::imshow("camera", image);
        if (cv::waitKey(1) > 0)
            return image;
    }
}

