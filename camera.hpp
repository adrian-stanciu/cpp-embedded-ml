#pragma once

#include <optional>

#include <fmt/core.h>
#include <opencv2/opencv.hpp>

namespace ic {
    struct Camera {
        static constexpr const char *GstreamerPipeline{R"(
            libcamerasrc !
            video/x-raw,
            width=(int)640,
            height=(int)480,
            framerate=(fraction)10/1 !
            videoconvert !
            appsink
        )"};

        Camera() : camera(GstreamerPipeline, cv::CAP_GSTREAMER) {}

        [[nodiscard]] bool is_open() const noexcept { return camera.isOpened(); }

        [[nodiscard]] cv::Mat read_image() noexcept
        {
            cv::Mat image;
            camera >> image;
            return image;
        }

    private:
        cv::VideoCapture camera;
    };

    [[nodiscard]] std::optional<cv::Mat> retrieve_image(const char *image_path);
}
