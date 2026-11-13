#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
using namespace cv;
using namespace std;

// 填充连通域内部孔洞
static void fillHoles(Mat& mask)
{
    vector<vector<Point>> contours;
    findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    drawContours(mask, contours, -1, Scalar(255), FILLED);
}

int main(int argc, char** argv)
{
    // 默认读取项目目录下的 test2.jpg，也可以从命令行传入路径
    const string inputPath = (argc > 1) ? argv[1] : "B:/Cyuyan/机器视觉/Project2/test2.jpg";
    const string outputPrefix =
        (argc > 2) ? argv[2] : "C:/Users/ASUS/Desktop/coin_result";

    Mat src = imread(inputPath, IMREAD_COLOR);
    if (src.empty()) {
        cerr << "Cannot read image: " << inputPath << endl;
        return -1;
    }

    // 1. 灰度化并滤波
    Mat gray, blurred;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    // 2. 硬币是亮目标，直接使用正向 Otsu
    Mat binary;
    double thresholdValue = threshold(
        blurred, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // 3. 闭运算连接缺口，开运算去掉小噪点
    Mat clean, kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(binary, clean, MORPH_CLOSE, kernel);
    morphologyEx(clean, clean, MORPH_OPEN, kernel);
    fillHoles(clean);

    // 4. 连通域标记
    Mat labels, stats, centroids;
    int labelCount = connectedComponentsWithStats(
        clean, labels, stats, centroids, 8, CV_32S);

    Mat coloredLabels = Mat::zeros(src.size(), CV_8UC3);
    Mat acceptedMask = Mat::zeros(src.size(), CV_8UC1);
    Mat result = src.clone();

    int minArea = max(150, static_cast<int>(src.total() * 0.003));
    int found = 0;

    for (int i = 1; i < labelCount; ++i) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int w = stats.at<int>(i, CC_STAT_WIDTH);
        int h = stats.at<int>(i, CC_STAT_HEIGHT);

        // 过滤小噪点和接触图像边界的白边
        bool touchesBorder = x <= 0 || y <= 0 ||
            x + w >= src.cols || y + h >= src.rows;
        if (area < minArea || touchesBorder) {
            continue;
        }

        ++found;

        // 提取当前硬币的掩膜
        Mat coinMask;
        compare(labels, i, coinMask, CMP_EQ);

        // 给每个硬币填不同颜色
        RNG rng(i * 7919 + 12345);
        Vec3b color(
            static_cast<uchar>(rng.uniform(50, 256)),
            static_cast<uchar>(rng.uniform(50, 256)),
            static_cast<uchar>(rng.uniform(50, 256)));

        coloredLabels.setTo(color, coinMask);
        acceptedMask.setTo(255, coinMask);

        // 绘制外接矩形
        Rect box(x, y, w, h);
        rectangle(result, box, Scalar(0, 255, 0), 2);

        // 用拟合椭圆中心作为硬币中心
        Point2f center(x + w * 0.5f, y + h * 0.5f);
        vector<vector<Point>> contours;
        findContours(coinMask.clone(), contours,
            RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            if (contours.front().size() >= 5) {
                center = fitEllipse(contours.front()).center;
            }
            else {
                float radius = 0;
                minEnclosingCircle(contours.front(), center, radius);
            }
        }

        circle(result, center, 3, Scalar(0, 0, 255), FILLED);
        drawMarker(result, center, Scalar(0, 0, 255),
            MARKER_CROSS, 12, 2, LINE_AA);

        // 绘制编号
        string id = to_string(found);
        putText(result, id, Point(x, max(y - 8, 18)),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 3);
        putText(result, id, Point(x, max(y - 8, 18)),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
    }

    // 5. 保存结果
    imwrite(outputPrefix + "_binary.png", binary);
    imwrite(outputPrefix + "_clean_binary.png", clean);
    imwrite(outputPrefix + "_labels.png", coloredLabels);
    imwrite(outputPrefix + "_components_mask.png", acceptedMask);
    imwrite(outputPrefix + "_result.png", result);

    cout << "Otsu threshold: " << thresholdValue << endl;
    cout << "Coins found: " << found << endl;

    // 6. 显示完整中间过程
    imshow("0 - Source", src);
    imshow("1 - Gray", gray);
    imshow("2 - Gaussian Blur", blurred);
    imshow("3 - Binary", binary);
    imshow("4 - Clean Mask", clean);
    imshow("5 - Colored Labels", coloredLabels);
    imshow("6 - Final Result", result);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
