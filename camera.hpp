#pragma once

#include <fmt/core.h>
#include <opencv2/opencv.hpp>

struct Camera {
    static constexpr auto Width{640};
    static constexpr auto Height{480};
    static constexpr auto FrameRate{30};

    [[nodiscard]] static std::string gstreamer_pipeline(int width, int height, int framerate)
    {
        return fmt::format(R"(
            libcamerasrc !
            video/x-raw,
            width=(int){:d},
            height=(int){:d},
            framerate=(fraction){:d}/1 !
            videoconvert !
            appsink)",
            width,
            height,
            framerate);
    }

    Camera() : camera(gstreamer_pipeline(Width, Height, FrameRate), cv::CAP_GSTREAMER) {}

    [[nodiscard]] bool is_open() const noexcept
    {
        return camera.isOpened();
    }

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

