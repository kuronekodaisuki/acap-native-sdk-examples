#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "Marker.h"

#define MARGIN  1
#define FONT_SCALE 0.9
#define FONT_THICKNESS  2

const cv::Scalar RED(0, 0, 255);
const cv::Scalar YELLO(0, 255, 255);
const cv::Scalar BLACK(0, 0, 0);

Marker::Marker(TYPE type, float size)
{
    _markerSize = size;
    _markerCorner.resize(4);
    // Marker corners
    _corners.push_back(cv::Point3f(-size / 2, size / 2, 0));
    _corners.push_back(cv::Point3f(size / 2, size / 2, 0));
    _corners.push_back(cv::Point3f(size / 2, -size / 2, 0));
    _corners.push_back(cv::Point3f(-size / 2, -size / 2, 0));

    switch (type)
    {
    case MARKER_4X4:
        _dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
        break;
    case MARKER_5X5:
        _dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
        break;
    case MARKER_6X6:
        _dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100);
        break;
    case MARKER_7X7:
        _dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_100);
        break;
    }

    // Refine with sub-pixel�@https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
    _detectorParams = cv::aruco::DetectorParameters::create();
    _detectorParams->cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
}

/// <summary>
/// �J�����p�����[�^�̐ݒ�
/// </summary>
/// <param name="YamlFilepath">YAML�t�@�C���p�X</param>
/// <returns></returns>
bool Marker::LoadCameraParameters(const char* filepath)
{
    cv::FileStorage parameters;
    try
    {
        if (parameters.open(filepath, cv::FileStorage::FORMAT_XML))
        {
            parameters["intrinsic"] >> _cameraMatrix;
            parameters["distortion"] >> _distCoeffs;
            return true;
        }
    }
    catch (std::exception e)
    {
        printf("FAILED to load %s\n", filepath);
    }
    return false;
}

bool Marker::Detect(cv::Mat& image)
{
    // Detect markers
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedImgPoints;
    // Detect markers
    cv::aruco::detectMarkers(image, _dictionary, markerCorners, _markerIdx, _detectorParams, rejectedImgPoints);

    cv::aruco::drawDetectedMarkers(image, markerCorners, _markerIdx);


    _rvecs.clear();
    _tvecs.clear();
    for (int i = 0; i < _markerIdx.size(); i++)
    {
        cv::Vec3d rvec, tvec;

        // Estimate pose by solvePnP
        cv::solvePnP(_corners, markerCorners[i], _cameraMatrix, _distCoeffs, rvec, tvec, false, cv::SolvePnPMethod::SOLVEPNP_IPPE_SQUARE);
        _rvecs.push_back(rvec);
        _tvecs.push_back(tvec);

        _markerCorner[0] = cv::Point(markerCorners[i][0]);
        _markerCorner[1] = cv::Point(markerCorners[i][1]);
        _markerCorner[2] = cv::Point(markerCorners[i][2]);
        _markerCorner[3] = cv::Point(markerCorners[i][3]);
        cv::fillPoly(image, _markerCorner, BLACK);

//#ifdef DRAW_AXIS
        cv::aruco::drawAxis(image, _cameraMatrix, _distCoeffs, rvec, tvec, 0.1f);
//#endif

    }
    return 0 < _markerIdx.size();
}

/// <summary>
/// Get pixel value
/// </summary>
/// <param name="image"></param>
/// <param name="x"></param>
/// <param name="y"></param>
/// <param name="i"></param>
/// <returns></returns>
cv::Vec3b Marker::Pixel(cv::Mat& image, cv::Point3f pos, int i)
{
    std::vector<cv::Point3f> points{ pos };
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(points, _rvecs[i], _tvecs[i], _cameraMatrix, _distCoeffs, projectedPoints);

    cv::Vec3b pixel = image.at<cv::Vec3b>(projectedPoints[0]);
    //cv::circle(image, projectedPoints[0], 3, BLACK, 2);

    return pixel;
}