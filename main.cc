#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>

enum Shape {
    kSquare,
    kCircle
};

struct pointComp {
    bool operator()(const cv::Point2d& lhs, const cv::Point2d& rhs) const {
        return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
    }
};

// Returns cropped image copy
cv::Mat crop(const cv::Mat& img, const Shape shape = Shape::kCircle) {
    cv::Mat img_crop;
    if (shape == Shape::kSquare) {
        int width = img.size().width;
        int height = img.size().height;
        int pad_length = std::abs(width - height) / 2;
        if (width < height) {
            img_crop = img.rowRange(pad_length, height - pad_length);
        } else {
            img_crop = img.colRange(pad_length, width - pad_length);
        }
        img_crop = img_crop.clone();
    } else if (shape == Shape::kCircle) {
        img_crop = crop(img, Shape::kSquare);
        cv::Mat mask = cv::Mat::zeros(img_crop.size(), img_crop.type());
        cv::circle(mask, mask.size() / 2, mask.size().width / 2, cv::Scalar(255, 255, 255), -1, 16);
        cv::Mat temp;
        img_crop.copyTo(temp, mask);
        img_crop = temp;
    }
    return img_crop;
}

double getDistance(cv::Scalar color1, cv::Scalar color2) {
    return cv::norm(color1 - color2);
}

double getLineMetric(cv::LineIterator line, cv::Scalar color) {
    double init = 0;
    for (int i = 0; i < line.count; ++i, ++line) {
        init += getDistance(**line, color);
    }
    return init;
}

cv::Mat stringArt(const cv::Mat& img, const uint32_t string_thickness = 3,
                  const cv::Scalar bgcolor = cv::Scalar(255, 255, 255),
                  const cv::Scalar color = cv::Scalar(0, 0, 0),
                  const uint16_t nail_count = 200) {
    // Initialize images
    cv::Mat roi = crop(img, Shape::kCircle);
    cv::Mat img_art = cv::Mat::zeros(roi.size(), roi.type());

    const double radius = img_art.size().width / 2 - 1;
    const double angle_step = M_PI * 2 / nail_count;
    cv::circle(img_art, img_art.size() / 2, radius, bgcolor, -1, 16);

    // Initialize nail positions
    std::vector<cv::Point2d> nails;
    for (int i = 0; i < nail_count; ++i) {
        std::complex<double> temp = std::polar(radius, angle_step * i);
        cv::Point2d point(temp.imag(), temp.real());
        point += (cv::Point2d)img_art.size() / 2;
        nails.push_back(point);
    }

    // debug
    cv::imshow("roi", roi);

    // Optimize strings
    // std::set<LineSegment> lines;
    cv::Point2d current_nail = nails[0];
    for (int i = 0; i < 1000; ++i) {
        double candidate_distance = std::numeric_limits<double>::infinity();
        for (cv::Point2d nail : nails) {
            cv::LineIterator line(roi, current_nail, nail);
            double temp = getLineMetric(line, color);
            if (temp < candidate_distance) {
                candidate_distance = temp;
            }
        }

        // debug
        cv::imshow("art", img_art);
        cv::waitKey(1);
    }

    return img_art;
}

int main(int argc, char const* argv[]) {
    // Initialize filename
    std::string filename;
    if (argc == 2) {
        filename = argv[1];
    } else {
        std::cout << "Enter filename: ";
        std::cin >> filename;
    }

    // Open image
    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
    if (!img.data) return 1;

    // Create string art
    cv::Mat img_art = stringArt(img);

    // Show image
    // cv::imshow("img", img);
    // cv::imshow("img_art", img_art);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}