#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // 默认图片路径，也可以从命令行传入
    const string path = (argc > 1) ? argv[1]
        : "B:/Cyuyan/机器视觉/Project3/test3.jpg";

    Mat src = imread(path, IMREAD_COLOR);
    if (src.empty()) {
        cerr << "Cannot read image: " << path << endl;
        return -1;
    }

    // 1. 灰度化
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // 2. 高斯滤波，减少噪声
    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    // 3. 孔是暗目标，使用反相 Otsu 二值化
    Mat binary;
    threshold(blurred, binary, 0, 255,
        THRESH_BINARY_INV | THRESH_OTSU);

    // 4. 连通域标记
    Mat labels, stats, centroids;
    int labelCount = connectedComponentsWithStats(
        binary, labels, stats, centroids, 8, CV_32S);

    // 彩色连通域图和最终结果图
    Mat coloredLabels = Mat::zeros(src.size(), CV_8UC3);
    Mat result = src.clone();

    int found = 0;

    // 根据图像面积设置孔的面积范围
    const int minArea = static_cast<int>(src.total() * 0.002);
    const int maxArea = static_cast<int>(src.total() * 0.020);

    for (int i = 1; i < labelCount; ++i) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int w = stats.at<int>(i, CC_STAT_WIDTH);
        int h = stats.at<int>(i, CC_STAT_HEIGHT);

        // 只保留完整、位于右侧、面积合适的孔
        bool complete = x > 0 && y > 0 &&
            x + w < src.cols && y + h < src.rows;
        bool onRight = centroids.at<double>(i, 0) > src.cols * 0.40;

        if (!complete || !onRight ||
            area < minArea || area > maxArea) {
            continue;
        }

        ++found;

        // 提取当前孔的掩膜并填充颜色
        Mat holeMask;
        compare(labels, i, holeMask, CMP_EQ);

        RNG rng(i * 7919 + 12345);
        Vec3b color(
            static_cast<uchar>(rng.uniform(50, 256)),
            static_cast<uchar>(rng.uniform(50, 256)),
            static_cast<uchar>(rng.uniform(50, 256)));

        coloredLabels.setTo(color, holeMask);

        // 绘制外接圆、中心点和编号
        Point center(
            cvRound(centroids.at<double>(i, 0)),
            cvRound(centroids.at<double>(i, 1)));

        int radius = max(w, h) / 2 + 4;

        circle(result, center, radius, Scalar(0, 255, 0), 2);
        circle(result, center, 3, Scalar(0, 0, 255), FILLED);
        putText(result, to_string(found),
            Point(center.x - 8, center.y - radius - 8),
            FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(0, 255, 255), 2);

        cout << "Hole " << found
            << ": center=(" << center.x << ", " << center.y
            << "), area=" << area << endl;
    }

    cout << "Complete holes found: " << found << endl;

    // 5. 显示中间过程和最终结果，不保存图片
    imshow("0 - Source", src);
    imshow("1 - Gray", gray);
    imshow("2 - Gaussian Blur", blurred);
    imshow("3 - Binary", binary);
    imshow("4 - Connected Components", coloredLabels);
    imshow("5 - Final Result", result);

    waitKey(0);
    destroyAllWindows();
    return 0;
}