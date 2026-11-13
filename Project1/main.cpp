#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {

    // 填充连通域内部孔洞，避免反光或内部亮区影响连通域
    void fillHoles(cv::Mat& mask) {
        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(
            mask.clone(),
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE);

        cv::drawContours(
            mask,
            contours,
            -1,
            cv::Scalar(255),
            cv::FILLED);
    }

}  // namespace

int main(int argc, char** argv) {
    // 输入图片路径
    const std::string inputPath =
        (argc > 1)
        ? argv[1]
        : "B:/Cyuyan/机器视觉/Project1/test1.jpg";

    // 输出图片路径前缀
    const std::string outputPrefix =
        (argc > 2)
        ? argv[2]
        : "C:/Users/ASUS/Desktop/paperclip_result";

    std::cout << "正在读取图片: "
        << inputPath << std::endl;

    // 读取彩色原图
    cv::Mat image =
        cv::imread(inputPath, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "图片读取失败: "
            << inputPath << std::endl;
        return -1;
    }

    std::cout << "图片读取成功，尺寸: "
        << image.cols << " x "
        << image.rows << std::endl;

    // 1. 转为灰度图
    cv::Mat gray;
    cv::cvtColor(
        image,
        gray,
        cv::COLOR_BGR2GRAY);

    // 2. 高斯滤波
    cv::Mat blurred;
    cv::GaussianBlur(
        gray,
        blurred,
        cv::Size(5, 5),
        0.0);

    // 3. Otsu 二值化
    // 图片是亮背景、暗目标，因此必须使用 THRESH_BINARY_INV
    cv::Mat binary;

    const double thresholdValue = cv::threshold(
        blurred,
        binary,
        0.0,
        255.0,
        cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    std::cout << "Otsu 阈值: "
        << thresholdValue << std::endl;

    // 4. 形态学去噪
    cv::Mat cleanMask;

    const cv::Mat kernel =
        cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(5, 5));

    cv::morphologyEx(
        binary,
        cleanMask,
        cv::MORPH_OPEN,
        kernel);

    cv::morphologyEx(
        cleanMask,
        cleanMask,
        cv::MORPH_CLOSE,
        kernel);

    // 填充回形针内部孔洞
    fillHoles(cleanMask);

    // 5. 连通域标记，8 邻域
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    const int labelCount =
        cv::connectedComponentsWithStats(
            cleanMask,
            labels,
            stats,
            centroids,
            8,
            CV_32S);

    if (labelCount <= 1) {
        std::cerr << "没有找到前景连通域。"
            << std::endl;
        return -1;
    }

    // 过滤顶部边框和小噪点
    const int minArea = std::max(
        200,
        static_cast<int>(
            image.total() * 0.004));

    cv::Mat coloredLabels =
        cv::Mat::zeros(
            image.size(),
            CV_8UC3);

    cv::Mat acceptedMask =
        cv::Mat::zeros(
            image.size(),
            CV_8UC1);

    cv::Mat result =
        image.clone();

    int acceptedCount = 0;

    const cv::Scalar boxColor(
        0,
        255,
        0);

    // 标签 0 是背景，从标签 1 开始处理
    for (int label = 1;
        label < labelCount;
        ++label) {

        const int area =
            stats.at<int>(
                label,
                cv::CC_STAT_AREA);

        if (area < minArea) {
            continue;
        }

        ++acceptedCount;

        const int left =
            stats.at<int>(
                label,
                cv::CC_STAT_LEFT);

        const int top =
            stats.at<int>(
                label,
                cv::CC_STAT_TOP);

        const int width =
            stats.at<int>(
                label,
                cv::CC_STAT_WIDTH);

        const int height =
            stats.at<int>(
                label,
                cv::CC_STAT_HEIGHT);

        const cv::Rect box(
            left,
            top,
            width,
            height);

        // 提取当前连通域
        cv::Mat componentMask;
        cv::compare(
            labels,
            label,
            componentMask,
            cv::CMP_EQ);

        // 生成随机颜色
        cv::RNG rng(
            label * 7919 + 12345);

        const cv::Vec3b color(
            static_cast<uchar>(
                rng.uniform(50, 256)),
            static_cast<uchar>(
                rng.uniform(50, 256)),
            static_cast<uchar>(
                rng.uniform(50, 256)));

        coloredLabels.setTo(
            color,
            componentMask);

        acceptedMask.setTo(
            255,
            componentMask);

        // 绘制外接矩形
        cv::rectangle(
            result,
            box,
            boxColor,
            2);

        // 默认使用外接矩形中心
        cv::Point2f center(
            static_cast<float>(
                box.x + box.width * 0.5),
            static_cast<float>(
                box.y + box.height * 0.5));

        // 使用拟合椭圆中心
        std::vector<std::vector<cv::Point>>
            componentContours;

        cv::findContours(
            componentMask.clone(),
            componentContours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE);

        if (!componentContours.empty()) {
            const std::vector<cv::Point>& contour =
                componentContours.front();

            if (contour.size() >= 5) {
                const cv::RotatedRect fitted =
                    cv::fitEllipse(contour);

                center = fitted.center;
            }
            else {
                float radius = 0.0F;

                cv::minEnclosingCircle(
                    contour,
                    center,
                    radius);
            }
        }

        // 绘制中心点
        cv::circle(
            result,
            center,
            3,
            cv::Scalar(0, 0, 255),
            cv::FILLED,
            cv::LINE_AA);

        cv::drawMarker(
            result,
            center,
            cv::Scalar(0, 0, 255),
            cv::MARKER_CROSS,
            12,
            2,
            cv::LINE_AA);

        // 绘制编号
        const std::string id =
            std::to_string(acceptedCount);

        const int font =
            cv::FONT_HERSHEY_SIMPLEX;

        const double fontScale = 0.65;

        int baseline = 0;

        const cv::Size textSize =
            cv::getTextSize(
                id,
                font,
                fontScale,
                2,
                &baseline);

        const int textY =
            (top > textSize.height + 8)
            ? top - 8
            : top + textSize.height + 4;

        const cv::Point textOrigin(
            left,
            textY);

        // 黑色描边
        cv::putText(
            result,
            id,
            textOrigin,
            font,
            fontScale,
            cv::Scalar(0, 0, 0),
            3,
            cv::LINE_AA);

        // 黄色编号
        cv::putText(
            result,
            id,
            textOrigin,
            font,
            fontScale,
            cv::Scalar(0, 255, 255),
            2,
            cv::LINE_AA);
    }

    // 6. 保存结果
    const std::string binaryPath =
        outputPrefix + "_binary.png";

    const std::string cleanPath =
        outputPrefix + "_clean_binary.png";

    const std::string labelsPath =
        outputPrefix + "_labels.png";

    const std::string maskPath =
        outputPrefix + "_components_mask.png";

    const std::string resultPath =
        outputPrefix + "_result.png";

    bool saved = true;

    saved &= static_cast<bool>(
        cv::imwrite(binaryPath, binary));

    saved &= static_cast<bool>(
        cv::imwrite(cleanPath, cleanMask));

    saved &= static_cast<bool>(
        cv::imwrite(labelsPath, coloredLabels));

    saved &= static_cast<bool>(
        cv::imwrite(maskPath, acceptedMask));

    saved &= static_cast<bool>(
        cv::imwrite(resultPath, result));

    if (!saved) {
        std::cerr << "部分结果图片保存失败。"
            << std::endl;
    }

    std::cout << "原始连通域数量: "
        << labelCount - 1 << std::endl;

    std::cout << "筛选后连通域数量: "
        << acceptedCount << std::endl;

    std::cout << "结果图: "
        << resultPath << std::endl;

    // 7. 显示原图和结果
    cv::namedWindow(
        "0 - Original Image",
        cv::WINDOW_NORMAL);

    cv::imshow(
        "0 - Original Image",
        image);

    cv::namedWindow(
        "1 - Binary",
        cv::WINDOW_NORMAL);

    cv::imshow(
        "1 - Binary",
        binary);

    cv::namedWindow(
        "2 - Clean Binary",
        cv::WINDOW_NORMAL);

    cv::imshow(
        "2 - Clean Binary",
        cleanMask);

    cv::namedWindow(
        "3 - Colored Labels",
        cv::WINDOW_NORMAL);

    cv::imshow(
        "3 - Colored Labels",
        coloredLabels);

    cv::namedWindow(
        "4 - Final Result",
        cv::WINDOW_NORMAL);

    cv::imshow(
        "4 - Final Result",
        result);

    std::cout << "按任意键关闭窗口并退出..."
        << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}