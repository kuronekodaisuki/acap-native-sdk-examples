#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "Flood.h"

// Scan area Width:25cm AHeight:15cm
#define WIDTH  0.25f
#define HEIGHT 0.15f
// Offset of scan area from marker
#define OFFSET 0.00f

#define FONT_SCALE 0.9
#define FONT_THICKNESS  2

const cv::Scalar RED(0, 0, 255);
const cv::Scalar YELLO(0, 255, 255);
const cv::Scalar BLACK(0, 0, 0);

/// <summary>
/// Constructor
/// </summary>
/// <param name="type">type of marker</param>
/// <param name="size">marker size</param>
Flood::Flood(TYPE type, float size):  Marker(type, size)
{
    // 検出領域(座標系はY軸が上方向)
    _scanArea.push_back(cv::Point3f(-HEIGHT / 2, WIDTH / 2 - OFFSET, 0));         // 左上
    _scanArea.push_back(cv::Point3f(-HEIGHT / 2, WIDTH / 2 - OFFSET, -HEIGHT));   // 左下
    _scanArea.push_back(cv::Point3f(-HEIGHT / 2, -WIDTH / 2 - OFFSET, 0));        // 右上
    _scanArea.push_back(cv::Point3f(-HEIGHT / 2, -WIDTH / 2 - OFFSET, -HEIGHT));  // 右下

    // 環境光参照
    _scanArea.push_back(cv::Point3f(0, size * 0.6f, 0));
    _scanArea.push_back(cv::Point3f(0, -size * 0.6f, 0));
}

/// <summary>
/// Scan area
/// </summary>
/// <param name="image"></param>
void Flood::Scan(cv::Mat& image)
{
    puts("[色相 彩度]");
    cv::Point3f pos = _scanArea[0]; // 左上
    pos.z -= HEIGHT / 6;
    for (int y = 0; y < 3; y++)
    {
        pos.y = WIDTH / 2 - OFFSET; // 左端
        for (int x = 0; x < 5; x++)
        {
            pos.y -= WIDTH / 10;
            cv::Vec3b pixel = Pixel(image, pos);
            printf("[%03d %03d] ", pixel[0], pixel[1]);
        }
        puts("");
        pos.z -= HEIGHT / 3;
    }
    // マーカー横の環境光参照点
    //cv::Vec3b pixel = Pixel(image, _scanArea[4]);
    //printf("[%03d %03d %03d]\n", pixel[0], pixel[1], pixel[2]);
}

/// <summary>
/// Detect flood marker
/// </summary>
/// <param name="image"></param>
/// <returns></returns>
bool Flood::Detect(cv::Mat& image)
{
    cv::Point2f leftTop;
    cv::Point2f rightTop;
    cv::Point2f leftBottom;
    cv::Point2f rightBottom;

    if (Marker::Detect(image))
    {
        for (int i = 0; i < _markerIdx.size(); i++)
        {
            // 検出領域(座標系はY軸が上方向)
            std::vector<cv::Point2f> projectedPoints;
            cv::projectPoints(_scanArea, _rvecs[i], _tvecs[i], _cameraMatrix, _distCoeffs, projectedPoints);

            leftTop = projectedPoints[0];
            rightTop = projectedPoints[1];
            leftBottom = projectedPoints[2];
            rightBottom = projectedPoints[3];

            cv::line(image, leftTop, rightTop, RED, 2);
            cv::line(image, leftBottom, rightBottom, RED, 2);
            cv::line(image, leftTop, leftBottom, RED, 2);
            cv::line(image, rightTop, rightBottom, RED, 2);
            cv::putText(image, "1", leftTop, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, FONT_SCALE, RED, FONT_THICKNESS);
            cv::putText(image, "2", rightTop, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, FONT_SCALE, RED, FONT_THICKNESS);
            cv::putText(image, "3", leftBottom, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, FONT_SCALE, RED, FONT_THICKNESS);
            cv::putText(image, "4", rightBottom, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, FONT_SCALE, RED, FONT_THICKNESS);
        }
        return true;
    }
    else
        return false;
}
