#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {
    // 【修复版孔洞填充】专门填充mask内部白色孔洞
    void fillHoles(cv::Mat& mask)
    {
        // mask: 前景=255(白)，背景=0(黑)
        cv::Mat holes = cv::Mat::zeros(mask.size(), CV_8UC1);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        // RETR_CCOMP：同时提取外层轮廓+内层孔洞轮廓
        cv::findContours(mask.clone(), contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++)
        {
            // hierarchy[i][3] != -1 → 这是内层孔洞轮廓
            if (hierarchy[i][3] != -1)
            {
                cv::drawContours(holes, contours, i, cv::Scalar(255), cv::FILLED);
            }
        }
        // 把孔洞填到原mask上
        mask = mask | holes;
    }
}  // namespace

int main(int argc, char** argv) {
    // 输入图片路径
    const std::string inputPath =
        (argc > 1)
        ? argv[1]
        : "B:/Cyuyan/机器视觉/Project2/test2.jpg";

    // 输出图片路径前缀
    const std::string outputPrefix =
        (argc > 2)
        ? argv[2]
        : "C:/Users/ASUS/Desktop/paperclip_result";

    std::cout << "正在读取图片: "
        << inputPath << std::endl;

    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "图片读取失败: " << inputPath << std::endl;
        return -1;
    }
    std::cout << "图片读取成功，尺寸: " << image.cols << " x " << image.rows << std::endl;

    // 1.灰度+高斯
    cv::Mat gray, blurred;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0.0);

    // 2. Otsu二值化：亮背景、暗硬币 → INV之后：硬币=黑(0)，背景=白(255)
    cv::Mat binary;
    const double thresholdValue = cv::threshold(
        blurred,
        binary,
        0.0,
        255.0,
        cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    std::cout << "Otsu 阈值: " << thresholdValue << std::endl;

    cv::Mat binary_inv;
    cv::bitwise_not(binary, binary_inv);

    // 3.形态学：先闭后开，先把反光洞合上，再去小噪点，核不要太大，Size(3,3)！
    cv::Mat cleanMask;
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary_inv, cleanMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(cleanMask, cleanMask, cv::MORPH_OPEN, kernel);

    // 填充硬币内部反光孔洞
    fillHoles(cleanMask);

    // 4.连通域
    cv::Mat labels, stats, centroids;
    const int labelCount = cv::connectedComponentsWithStats(cleanMask, labels, stats, centroids, 8, CV_32S);
    if (labelCount <= 1) {
        std::cerr << "没有找到前景连通域。" << std::endl;
        return -1;
    }

    // 最小面积，你硬币比较大，这里可以改成 150，也可以自动比例
    const int minArea = std::max(150, static_cast<int>(image.total() * 0.003));
    cv::Mat coloredLabels = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::Mat acceptedMask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::Mat result = image.clone();
    int acceptedCount = 0;
    const cv::Scalar boxColor(0, 255, 0);

    for (int label = 1; label < labelCount; ++label)
    {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < minArea) continue;

        acceptedCount++;
        const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const cv::Rect box(left, top, width, height);

        cv::Mat componentMask;
        cv::compare(labels, label, componentMask, cv::CMP_EQ);

        cv::RNG rng(label * 7919 + 12345);
        const cv::Vec3b color(
            static_cast<uchar>(rng.uniform(50, 256)),
            static_cast<uchar>(rng.uniform(50, 256)),
            static_cast<uchar>(rng.uniform(50, 256)));
        coloredLabels.setTo(color, componentMask);
        acceptedMask.setTo(255, componentMask);

        cv::rectangle(result, box, boxColor, 2);

        // 中心点
        cv::Point2f center(box.x + box.width * 0.5f, box.y + box.height * 0.5f);
        std::vector<std::vector<cv::Point>> componentContours;
        cv::findContours(componentMask.clone(), componentContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (!componentContours.empty())
        {
            auto& contour = componentContours.front();
            if (contour.size() >= 5)
            {
                cv::RotatedRect fitted = cv::fitEllipse(contour);
                center = fitted.center;
            }
            else
            {
                float radius;
                cv::minEnclosingCircle(contour, center, radius);
            }
        }
        cv::circle(result, center, 3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        cv::drawMarker(result, center, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 12, 2, cv::LINE_AA);

        // 编号文字
        std::string id = std::to_string(acceptedCount);
        int font = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.65;
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(id, font, fontScale, 2, &baseline);
        int textY = (top > textSize.height + 8) ? top - 8 : top + textSize.height + 4;
        cv::Point textOrigin(left, textY);
        cv::putText(result, id, textOrigin, font, fontScale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(result, id, textOrigin, font, fontScale, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }

    // 保存
    const std::string binaryPath = outputPrefix + "_binary.png";
    const std::string cleanPath = outputPrefix + "_clean_binary.png";
    const std::string labelsPath = outputPrefix + "_labels.png";
    const std::string maskPath = outputPrefix + "_components_mask.png";
    const std::string resultPath = outputPrefix + "_result.png";
    bool saved = true;
    saved &= static_cast<bool>(cv::imwrite(binaryPath, binary));
    saved &= static_cast<bool>(cv::imwrite(cleanPath, cleanMask));
    saved &= static_cast<bool>(cv::imwrite(labelsPath, coloredLabels));
    saved &= static_cast<bool>(cv::imwrite(maskPath, acceptedMask));
    saved &= static_cast<bool>(cv::imwrite(resultPath, result));
    if (!saved) std::cerr << "部分结果图片保存失败。" << std::endl;

    std::cout << "原始连通域数量: " << labelCount - 1 << std::endl;
    std::cout << "筛选后连通域数量: " << acceptedCount << std::endl;
    std::cout << "结果图: " << resultPath << std::endl;

    // 窗口显示
    cv::namedWindow("0 - Original Image", cv::WINDOW_NORMAL);
    cv::imshow("0 - Original Image", image);
    cv::namedWindow("1 - Binary", cv::WINDOW_NORMAL);
    cv::imshow("1 - Binary", binary);
    cv::namedWindow("2 - Clean Binary", cv::WINDOW_NORMAL);
    cv::imshow("2 - Clean Binary", cleanMask);
    cv::namedWindow("3 - Colored Labels", cv::WINDOW_NORMAL);
    cv::imshow("3 - Colored Labels", coloredLabels);
    cv::namedWindow("4 - Final Result", cv::WINDOW_NORMAL);
    cv::imshow("4 - Final Result", result);

    std::cout << "按任意键关闭窗口并退出..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
