#pragma once

#ifdef WIN32
#ifdef EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif
#else
#define API
#endif

#ifdef WIN32
#pragma warning(disable:4819)
#endif // DEBUG
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <map>
#include <iostream>
#include <fstream>
#include "MovingAvg.h"


#define OFFSET 0.08

#ifdef USE_KALMANFILTER
class KalmanFilter;
#endif

class API SnowDetector
{
public:
	enum MARKER_TYPE { MARKER_4X4, MARKER_5X5, MARKER_6X6, MARKER_7X7 };
	enum STATUS { OK, NG, LOST };

	SnowDetector();
	SnowDetector(MARKER_TYPE markerType, float markerSize, float poleLength = -1);

	bool Configure(const char* filepath, MARKER_TYPE markerType, float markerSize, float poleLength = -1);
	bool LoadCameraParameters(const char* filepath);

	float Detect(cv::Mat image);

	void Log(std::ofstream& output);

	void PoleLengthOffset(float offset) { _poleLengthOffset += offset; };

	cv::Mat& Deskew() { return _deskew; }

  STATUS GetStatus() { return _status; }

private:
	float _markerSize;
	float _poleLength;
	float _poleLengthOffset = 0;
	float _poleOffset = 0;
	std::vector<cv::Point3f> _corners;
	std::vector<cv::Point> _markerCorner;
	STATUS _status;
	float _depth;
	float _theta;

	typedef struct
	{
		STATUS status;
		float depth;
	}STATE;

	double _fps = 10;
	double _dt = 0.1; // Time step
	std::vector<STATE> _states;
	int _counter = 0;

	cv::KalmanFilter _kalmanFilter;

	std::map<int, MovingAvg> _average;	// Moving Average

	cv::Ptr<cv::aruco::Dictionary> _dictionary;
	cv::Ptr<cv::aruco::DetectorParameters> _detectorParams;
	cv::Mat _cameraMatrix, _distCoeffs, _grayscale, _deskew;
	std::vector<int> _markerIdx;
	std::vector<cv::Vec3d> _rvecs, _tvecs;
	std::vector<cv::Point3f> _axesPoints;

	void Initialize(MARKER_TYPE markerType, float markerSize, float poleLength);

	float estimateDepth(cv::Mat bilevel, cv::Mat& image, int index, float poleLength);

	void ShowMesh(cv::Mat image, cv::Vec3d& tvec, cv::Vec3d& rvec);

	void drawPole(cv::Mat& image, cv::Vec3d& tvec, cv::Vec3d& rvec, int thickness);
};

