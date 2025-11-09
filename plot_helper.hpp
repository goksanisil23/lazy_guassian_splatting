#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat plotLosses(const std::vector<float> &losses)
{
    constexpr int kWidth  = 1200;
    constexpr int kHeight = 800;

    cv::Mat plot(kHeight, kWidth, CV_8UC3, cv::Scalar(30, 30, 30));

    const float max_val = *std::max_element(losses.begin(), losses.end());
    const float min_val = *std::min_element(losses.begin(), losses.end());
    const float range   = std::max(1e-6f, max_val - min_val);

    const int n = static_cast<int>(losses.size());

    const auto xy = [&](const int i) -> cv::Point
    {
        const int x = static_cast<int>(std::lround(i * (kWidth - 1) / static_cast<float>(std::max(1, n - 1))));
        const int y = kHeight - 1 - static_cast<int>(std::lround((losses[i] - min_val) / range * (kHeight - 1)));
        return cv::Point(x, y);
    };

    // Draw line segments
    for (int i = 1; i < n; ++i)
        cv::line(plot, xy(i - 1), xy(i), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    // Draw points and loss values
    for (int i = 0; i < n; ++i)
    {
        const cv::Point p = xy(i);
        cv::circle(plot, p, 2, cv::Scalar(0, 200, 255), cv::FILLED, cv::LINE_AA);

        // Label every Nth point (to avoid clutter)
        if ((i == n - 1) || (i % 10 == 0))
        {
            const std::string text = cv::format("%.3f", losses[i]);
            cv::putText(plot,
                        text,
                        p + cv::Point(5, -5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.25,
                        cv::Scalar(255, 255, 255),
                        1,
                        cv::LINE_AA);
        }
    }

    cv::imshow("Training Loss", plot);
    cv::waitKey(1);

    return plot;
}