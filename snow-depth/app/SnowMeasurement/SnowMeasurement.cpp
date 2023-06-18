
#define _USE_MATH_DEFINES
#include <math.h>
//#include <Eigen/Dense>

#include "SnowMeasurement.h"
#include "RotateAndCrop.h"
#include "SnowDepth.h"

#ifdef WIN32
#pragma warning(disable: 4819)
#endif

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
//#include <opencv2/core/eigen.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <syslog.h>

#define MESH_SIZE   2
#define POLE_WIDTH  0.05f
#define MARGIN  1
#define FONT_SCALE 0.9
#define FONT_THICKNESS  2

const cv::Scalar RED(0, 0, 255);
const cv::Scalar YELLO(0, 255, 255);
const cv::Scalar BLACK(0, 0, 0);


SnowDetector::SnowDetector()
{
    _poleOffset = (float)OFFSET;
}

void initKalmanFilter(cv::KalmanFilter& KF, int nStates, int nMeasurements, int nInputs, double dt)
{
    KF.init(nStates, nMeasurements, nInputs, CV_64F); // init Kalman Filter
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5)); // set process noise
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-4)); // set measurement noise
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1)); // error covariance
    /* DYNAMIC MODEL */
    // [1 0 0 dt 0 0 dt2 0 0 0 0 0 0 0 0 0 0 0]
    // [0 1 0 0 dt 0 0 dt2 0 0 0 0 0 0 0 0 0 0]
    // [0 0 1 0 0 dt 0 0 dt2 0 0 0 0 0 0 0 0 0]
    // [0 0 0 1 0 0 dt 0 0 0 0 0 0 0 0 0 0 0]
    // [0 0 0 0 1 0 0 dt 0 0 0 0 0 0 0 0 0 0]
    // [0 0 0 0 0 1 0 0 dt 0 0 0 0 0 0 0 0 0]
    // [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
    // [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
    // [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]
    // [0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0 dt2 0 0]
    // [0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0 dt2 0]
    // [0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0 dt2]
    // [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0 0]
    // [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt 0]
    // [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 dt]
    // [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
    // [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
    // [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
    // position
    KF.transitionMatrix.at<double>(0, 3) = dt;
    KF.transitionMatrix.at<double>(1, 4) = dt;
    KF.transitionMatrix.at<double>(2, 5) = dt;
    KF.transitionMatrix.at<double>(3, 6) = dt;
    KF.transitionMatrix.at<double>(4, 7) = dt;
    KF.transitionMatrix.at<double>(5, 8) = dt;
    KF.transitionMatrix.at<double>(0, 6) = 0.5 * pow(dt, 2);
    KF.transitionMatrix.at<double>(1, 7) = 0.5 * pow(dt, 2);
    KF.transitionMatrix.at<double>(2, 8) = 0.5 * pow(dt, 2);
    // orientation
    KF.transitionMatrix.at<double>(9, 12) = dt;
    KF.transitionMatrix.at<double>(10, 13) = dt;
    KF.transitionMatrix.at<double>(11, 14) = dt;
    KF.transitionMatrix.at<double>(12, 15) = dt;
    KF.transitionMatrix.at<double>(13, 16) = dt;
    KF.transitionMatrix.at<double>(14, 17) = dt;
    KF.transitionMatrix.at<double>(9, 15) = 0.5 * pow(dt, 2);
    KF.transitionMatrix.at<double>(10, 16) = 0.5 * pow(dt, 2);
    KF.transitionMatrix.at<double>(11, 17) = 0.5 * pow(dt, 2);
    /* MEASUREMENT MODEL */
    // [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    // [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    // [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    // [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    // [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    // [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
    KF.measurementMatrix.at<double>(0, 0) = 1; // x
    KF.measurementMatrix.at<double>(1, 1) = 1; // y
    KF.measurementMatrix.at<double>(2, 2) = 1; // z
    KF.measurementMatrix.at<double>(3, 9) = 1; // roll
    KF.measurementMatrix.at<double>(4, 10) = 1; // pitch
    KF.measurementMatrix.at<double>(5, 11) = 1; // yaw
}

/*
cv::Mat euler2rot(double roll, double pitch, double yaw)
{
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    return cv::Mat(1, 3, CV_64F, q.matrix().data());
}
*/

cv::Vec3d rot2euler(const cv::Vec3d& rvec)
{
    cv::Mat rot;
    cv::Rodrigues(rvec, rot);
    float sy = sqrt(rot.at<double>(0, 0) * rot.at<double>(0, 0) + rot.at<double>(1, 0) * rot.at<double>(1, 0));

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(rot.at<double>(2, 1), rot.at<double>(2, 2));
        y = atan2(-rot.at<double>(2, 0), sy);
        z = atan2(rot.at<double>(1, 0), rot.at<double>(0, 0));
    }
    else
    {
        x = atan2(-rot.at<double>(1, 2), rot.at<double>(1, 1));
        y = atan2(-rot.at<double>(2, 0), sy);
        z = 0;
    }
    return cv::Vec3d(x, y, z);
}

/*
void updateKalmanFilter(cv::KalmanFilter& KF, cv::Mat& measurement,
    cv::Mat& translation_estimated, cv::Mat& rotation_estimated)
{
    // First predict, to update the internal statePre variable
    cv::Mat prediction = KF.predict();
    // The "correct" phase that is going to use the predicted value and our measurement
    cv::Mat estimated = KF.correct(measurement);
    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);
    // Estimated euler angles
    // Convert estimated quaternion to rotation matrix
    rotation_estimated = euler2rot(estimated.at<double>(9), estimated.at<double>(10), estimated.at<double>(11));
}

void fillMeasurements(cv::Mat& measurements,
    const cv::Vec3d& translation_measured, const cv::Vec3d& rotation_measured)
{
    // Convert rotation matrix to euler angles
    cv::Vec3d measured_eulers = rot2euler(rotation_measured);
    // Set measurement to predict
    measurements.at<double>(0) = translation_measured[0]; // x
    measurements.at<double>(1) = translation_measured[1]; // y
    measurements.at<double>(2) = translation_measured[2]; // z
    measurements.at<double>(3) = measured_eulers[0]; // roll
    measurements.at<double>(4) = measured_eulers[1]; // pitch
    measurements.at<double>(5) = measured_eulers[2]; // yaw
}
*/

void SnowDetector::Initialize(MARKER_TYPE markerType, float markerSize, float poleLength)
{
    _markerSize = markerSize;
    _poleLength = poleLength;
    _poleOffset = (float)OFFSET;
    // Marker corners
    _corners.push_back(cv::Point3f(-markerSize / 2, markerSize / 2, 0));
    _corners.push_back(cv::Point3f(markerSize / 2, markerSize / 2, 0));
    _corners.push_back(cv::Point3f(markerSize / 2, -markerSize / 2, 0));
    _corners.push_back(cv::Point3f(-markerSize / 2, -markerSize / 2, 0));

    switch (markerType)
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

    // Refine with sub-pixel https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
    _detectorParams = cv::aruco::DetectorParameters::create();
    _detectorParams->cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;

    //
    _axesPoints.push_back(cv::Point3f(_poleOffset - POLE_WIDTH, 0, 0));              //
    _axesPoints.push_back(cv::Point3f(_poleOffset + POLE_WIDTH, 0, 0));              //
    _axesPoints.push_back(cv::Point3f(_poleOffset - POLE_WIDTH, -poleLength, 0));    //
    _axesPoints.push_back(cv::Point3f(_poleOffset + POLE_WIDTH, -poleLength, 0));    //
    _axesPoints.push_back(cv::Point3f(_poleOffset, -poleLength / 2, 0));  //

    int nStates = 18; // the number of states
    int nMeasurements = 6; // the number of measured states
    int nInputs = 0; // the number of action control
    double dt = 0.1; // time between measurements (1/FPS)
    initKalmanFilter(_kalmanFilter, nStates, nMeasurements, nInputs, dt); // init function
    _markerCorner.resize(4);
    _states.resize(_fps * 5);
}

bool SnowDetector::Configure(const char* filepath, MARKER_TYPE markerType, float markerSize, float poleLength)
{
    Initialize(markerType, markerSize, poleLength);
    return LoadCameraParameters(filepath);
}

/// <summary>
///
/// </summary>
/// <param name="markerType"></param>
/// <param name="markerSize">(m)</param>
/// <param name="poleLength">(m)</param>
SnowDetector::SnowDetector(MARKER_TYPE markerType, float markerSize, float poleLength)
{
    Initialize(markerType, markerSize, poleLength);
}

/// <summary>
///
/// </summary>
/// <param name="YamlFilepath">YAML</param>
/// <returns></returns>
bool SnowDetector::LoadCameraParameters(const char* filepath)
{
    cv::FileStorage parameters;
    try
    {
        if (parameters.open(filepath, cv::FileStorage::FORMAT_XML))
        {
            parameters["intrinsic"] >> _cameraMatrix;
            parameters["distortion"] >> _distCoeffs;

            for (int i = 0; i < 9; i++)
              syslog(LOG_DEBUG, "intrinsic[%d]:%f", i, _cameraMatrix.at<double>(i));
            for (int i = 0; i < 5; i++)
              syslog(LOG_DEBUG, "distortion[%d]:%f", i, _distCoeffs.at<double>(i));
            return true;
        }
    }
    catch (std::exception e)
    {
        printf("FAILED to load %s\n", filepath);
    }
    return false;
}

/// <summary>
/// Output result
/// </summary>
/// <param name="output"></param>
void SnowDetector::Log(std::ofstream& output)
{
    const char* status;
    switch (_status)
    {
    case LOST:
        status = "LOST";
        break;

    case NG:
        status = "NG";
        break;

    default:
        status = "OK";
        break;
    }
    if (0 < _rvecs.size())
        output << status << "," << _depth << "," << _theta * 180 / M_PI << "," << _rvecs[0] << std::endl;
    else
        output << status << "," << _depth << "," << _theta * 180 / M_PI << "," << std::endl;
}

/// <summary>
/// Detect snow amount
/// </summary>
/// <param name="image"></param>
/// <returns></returns>
float SnowDetector::Detect(cv::Mat image)
{
    float depth = -1;
    _status = LOST;

    // Detect markers
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedImgPoints;
    // Detect markers
    cv::aruco::detectMarkers(image, _dictionary, markerCorners, _markerIdx, _detectorParams, rejectedImgPoints);
    for (int i = 0; i < _markerIdx.size(); i++)
    {
      syslog(LOG_INFO, "Marker:%d", _markerIdx[i]);
    }
#ifdef DRAW_MARKERS
    cv::aruco::drawDetectedMarkers(image, markerCorners, _markerIdx);
#endif

    cv::Point2f center;
    if (0 < markerCorners.size())
    {
        // binarize
        cv::Mat biLevel;
        cv::cvtColor(image, _grayscale, cv::COLOR_RGB2GRAY);
        //
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::erode(_grayscale, _grayscale, kernel);
        cv::threshold(_grayscale, biLevel, 0, 255, cv::THRESH_OTSU);

        _rvecs.clear();
        _tvecs.clear();
        for (int i = 0; i < _markerIdx.size(); i++)
        {
            cv::Vec3d rvec, tvec;

            // Estimate pose by solvePnP
            cv::solvePnP(_corners, markerCorners[i], _cameraMatrix, _distCoeffs, rvec, tvec, false, cv::SolvePnPMethod::SOLVEPNP_IPPE_SQUARE);

#ifdef USE_KALMANFILTER
            cv::Vec3d euler = rot2euler(rvec);

            // KalmanFilter
            _kalmanFilter.predict();

            cv::Mat measurement(6, 1, CV_64F);
            measurement.at<double>(0) = tvec[0];
            measurement.at<double>(1) = tvec[1];
            measurement.at<double>(2) = tvec[2];
            measurement.at<double>(3) = euler[0];
            measurement.at<double>(4) = euler[1];
            measurement.at<double>(5) = euler[2];

            cv::Mat estimated = _kalmanFilter.correct(measurement);

            cv::Vec3d rot = euler2rot(estimated.at<double>(9), estimated.at<double>(10), estimated.at<double>(11));
            std::cout << rvec << euler << std::endl;
#endif
            _rvecs.push_back(rvec);
            _tvecs.push_back(tvec);

            // Calculate snow amount
            if (_poleLength < 0)
            {
                depth = estimateDepth(biLevel, image, i, (float)_markerIdx[i] / 10 + _poleLengthOffset);
            }
            else
            {
                depth = estimateDepth(biLevel, image, i, _poleLength + _poleLengthOffset);
            }
            _markerCorner[0] = cv::Point(markerCorners[i][0]);
            _markerCorner[1] = cv::Point(markerCorners[i][1]);
            _markerCorner[2] = cv::Point(markerCorners[i][2]);
            _markerCorner[3] = cv::Point(markerCorners[i][3]);
            cv::fillPoly(image, _markerCorner, BLACK);
            center = (markerCorners[i][1] + markerCorners[i][3]) / 2;

#ifdef DRAW_AXIS
            cv::aruco::drawAxis(image, _cameraMatrix, _distCoeffs, rvec, tvec, 1);
#endif
        }
    }
    else
    {
        cv::fillPoly(image, _markerCorner, BLACK);
    }
    //
    if (_states.size() <= _counter)
    {
        float sum = 0;
        _counter = 0;
        for (size_t i = 0; i < _states.size(); i++)
        {
            if (_states[i].status == STATUS::OK)
            {
                sum += _states[i].depth;
                _counter++;
            }
        }
        _depth = sum / _counter;
        _counter = 0;

        // JSON
        SnowDepth snow(_depth);
        snow.Send(std::cout);
        std::cout << std::endl;
    }
    else
    {
        _states[_counter].depth = depth;
        _states[_counter].status = _status;
        _counter++;
    }
    //if (0 < _depth)
    {
        char buffer[16];
        sprintf(buffer, "%.0f", _depth);

        int baseline = 0;
        cv::Size textSize = getTextSize(buffer, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, FONT_SCALE, FONT_THICKNESS, &baseline);
        cv::Point textOrigin(center.x - textSize.width / 2, center.y + textSize.height / 2);
        cv::putText(image, buffer, textOrigin, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, FONT_SCALE, RED, FONT_THICKNESS);
        cv::putText(image, "cm", cv::Point(textOrigin.x + textSize.width, textOrigin.y), cv::FONT_HERSHEY_SIMPLEX, 0.7, RED, FONT_THICKNESS);
    }
    return _depth;
}

/// <summary>
/// Correct Z flip
/// </summary>
/// <param name="tvec"></param>
/// <param name="rvec"></param>
/// <returns>true when corrected</returns>
static cv::Vec3d FixRotate(cv::Vec3d T, cv::Vec3d rvec)
{
#if 0
    cv::Mat R1, R2, Rlast;
    cv::Rodrigues(rvec, Rlast);
    Eigen::Matrix3d eigMatLast;
    cv2eigen(Rlast, eigMatLast);
    Eigen::Quaterniond quatLast(eigMatLast);

    cv::Rodrigues(rvec, R1);
    Eigen::Matrix3d eigMat1;
    cv2eigen(R1, eigMat1);
    Eigen::Quaterniond quat1(eigMat1);

    cv::Rodrigues(rvec, R2);
    Eigen::Matrix3d eigMat2;
    cv2eigen(R2, eigMat2);
    Eigen::Quaterniond quat2(eigMat2);

    double dist1 = quatLast.angularDistance(quat1);
    double dist2 = quatLast.angularDistance(quat2);

    if (dist1 < dist2)
    {
        //marker.rvec = rvecsVec[0];
        //marker.tvec = tvecsVec[0];
        //solveIdx = 0;
    }
    if (dist1 > dist2)
    {
        //marker.rvec = rvecsVec[1];
        //marker.tvec = tvecsVec[1];
    }
#else

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    double swap[3][3] = {
        {1, 0, 0},
        {0, 0, 1},
        {0, 1, 0}
    };
    double flip[3][3] = {
        {1, -1, 1},
        {1, -1, 1},
        {-1, 1, -1}
    };

    cv::Mat SwapAxes(3, 3, R.type(), swap);
    cv::Mat FlipAxes(3, 3, R.type(), flip);

    std::cout << R;

    R = R * SwapAxes;
    std::cout << R;

    if (0 < R.at<double>(2, 2) && R.at<double>(2, 2) < 1)
    {
        R *= FlipAxes;
        std::cout << R << std::endl;

        //# Fixup: rotate along the plane spanned by camera's forward (Z) axis and vector to marker's position
        cv::Vec3d forward(0, 0, 1);
        cv::Vec3d tnorm = T / sqrt(T[0] * T[0] + T[1] * T[1] + T[2] * T[2]);

        cv::Vec3d axis = tnorm.cross(forward);

        double angle = -2 * acos(tnorm[0] * forward[0] + tnorm[1] * forward[1] + tnorm[2] * forward[2]);

        std::cout << angle << " " << axis << std::endl;
        std::cout << angle * axis << std::endl;

        cv::Rodrigues(angle * axis, R);
        R *= SwapAxes;
        cv::Rodrigues(R, rvec);

        std::cout << rvec << std::endl;
        return rvec;
    }
#endif
    return rvec;
}

/// <summary>
///
/// </summary>
/// <param name="grayscale"></param>
/// <param name="image"></param>
/// <param name="index"></param>
/// <param name="poleLength"></param>
/// <returns></returns>
float SnowDetector::estimateDepth(cv::Mat bilevel, cv::Mat& image, int index, float poleLength)
{
    syslog(LOG_INFO, "%s", __func__);
    float depth = -1;
    _status = OK;

    cv::Point2f leftTop;
    cv::Point2f rightTop;
    cv::Point2f leftBottom;
    cv::Point2f rightBottom;
    cv::Point2f center;

    //
    cv::Vec3d tvec = _tvecs[index];
    cv::Vec3d rvec = _rvecs[index];

    if (rvec[1] < 0)
    {
      syslog(LOG_INFO, "rvec[1]:%f", rvec[1]);
        //
        //rvec = FixRotate(tvec, rvec);

        _status = NG;
        //return -1;
    }
    _theta = sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]);
    syslog(LOG_INFO, "Theta:%f", _theta);

    //char buffer[200];
    //sprintf(buffer, "Theta:%f R:%f %f %f", _theta * 180 / M_PI, rvec[0], rvec[1], rvec[2]);
    //cv::putText(image, buffer, cv::Point2f(0, 60), cv::FONT_HERSHEY_PLAIN, 1.5, YELLO);

    auto find = _average.find(_markerIdx[index]);
    if (find == _average.end())
    {
        _average[_markerIdx[index]] = MovingAvg(tvec, rvec);
    }
    else
    {
        _average[_markerIdx[index]].Update(tvec, rvec);
    }

    //
    std::vector<cv::Point2f> projectedPoints;
    projectPoints(_axesPoints, rvec, tvec, _cameraMatrix, _distCoeffs, projectedPoints);

    leftTop = projectedPoints[0];
    rightTop = projectedPoints[1];
    leftBottom = projectedPoints[2];
    rightBottom = projectedPoints[3];
    center = projectedPoints[4];

    //cv::line(image, leftTop, rightTop, RED);
    //cv::line(image, leftBottom, rightBottom, RED);
    //cv::line(image, leftTop, leftBottom, RED);
    //cv::line(image, rightTop, rightBottom, RED);

    //
    double angle = (double)atan2((leftBottom.y - leftTop.y), (leftBottom.x - leftTop.x)) * 180 / M_PI;

    float width = sqrt((leftTop.x - rightTop.x) * (leftTop.x - rightTop.x) + (leftTop.y - rightTop.y) * (leftTop.y - rightTop.y));
    float height = sqrt((leftTop.x - leftBottom.x) * (leftTop.x - leftBottom.x) + (leftTop.y - leftBottom.y) * (leftTop.y - leftBottom.y));
    //
    _deskew = RotateAndCrop(_grayscale, center.x, center.y, angle - 90, (int)width * 2, (int)height);
    // deskew
    cv::Mat bi;
    double threshold = cv::threshold(_grayscale, bi, 0, 255, cv::THRESH_OTSU) * MARGIN;
#ifdef _DEBUIG
    std::cout << threshold << std::endl;
#endif
    //
    if (abs(90 - angle) < 10)
    {
        // TODO

        //
        float left = leftBottom.x;
        float right = rightBottom.x;
        float delta = (leftBottom.x - leftTop.x) / (leftBottom.y - leftTop.y);

        //
        for (int y = (int)leftBottom.y; y > leftTop.y; y--)
        {
            uchar* ptr = bilevel.ptr(y);
            //
            int value = 0, pixels = 0;

            for (int x = (int)left; x < right; x++, pixels++)
            {
                value += ptr[x];
            }
            value /= pixels;

            left -= delta;
            right -= delta;

            //
            if (value < threshold)
            {
                depth = std::round(poleLength * 100 * (leftBottom.y - y) / (leftBottom.y - leftTop.y));
                cv::line(image, cv::Point2f(leftBottom.x, (float)y), cv::Point2f(rightBottom.x, (float)y), RED, 3);

                //sprintf(buffer, "%.0fcm", depth);

                //cv::putText(image, buffer, cv::Point2f(rightBottom.x, (float)y), cv::FONT_HERSHEY_PLAIN, 2, RED);
                //cv::imwrite("snapshot.png", image);

                break;
            }
            //printf_s("Y:%d value: %d\n", y, value);
        }
    }
    else
    {
        printf("Skew:%f\n", angle);
    }
    //cv::imshow("deskew", _deskew);

    return depth;
}

/// <summary>
///
/// </summary>
/// <param name="image"></param>
void SnowDetector::ShowMesh(cv::Mat image, cv::Vec3d& tvec, cv::Vec3d& rvec)
{
    //
    std::vector<cv::Point3f> points;
    for (int x = -MESH_SIZE; x <= MESH_SIZE; x++)
    {
        points.push_back(cv::Point3f((float)x, -_poleLength, MESH_SIZE));
        points.push_back(cv::Point3f((float)x, -_poleLength, -MESH_SIZE));
    }
    for (int z = -MESH_SIZE; z <= MESH_SIZE; z++)
    {
        points.push_back(cv::Point3f(-MESH_SIZE, -_poleLength, (float)z));
        points.push_back(cv::Point3f(MESH_SIZE, -_poleLength, (float)z));
    }
    std::vector<cv::Point2f> projectedPoints;
    projectPoints(points, rvec, tvec, _cameraMatrix, _distCoeffs, projectedPoints);
    for (int i = 0; i < projectedPoints.size(); i += 2)
    {
        cv::line(image, projectedPoints[i], projectedPoints[i + 1], cv::Scalar(128, 255, 255));
    }
}

/// <summary>
///
/// </summary>
/// <param name="image"></param>
/// <param name="index"></param>
/// <param name="thickness"></param>
void SnowDetector::drawPole(cv::Mat& image, cv::Vec3d& tvec, cv::Vec3d& rvec, int thickness)
{
    int numPoints = (int)(_poleLength / 0.1f);
    // project axes points
    std::vector<cv::Point3f> axesPoints;

    //
    for (int i = 0; i < numPoints; i++)
    {
        axesPoints.push_back(cv::Point3f(_poleOffset, i * -0.1f, 0));
    }
    //
    axesPoints.push_back(cv::Point3f(_poleOffset, -_poleLength, 0));

    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(axesPoints, rvec, tvec, _cameraMatrix, _distCoeffs, projectedPoints);

    //
    cv::line(image, projectedPoints[0], projectedPoints[numPoints], RED, thickness);

    //
    for (int i = 0; i < numPoints; i += 2)
    {
        cv::line(image, projectedPoints[i], projectedPoints[i + 1], cv::Scalar(0, 255, 255), thickness);
    }
}
