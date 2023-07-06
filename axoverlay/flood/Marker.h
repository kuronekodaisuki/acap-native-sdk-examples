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
#include <map>
#include <iostream>
#include <fstream>

class API Marker
{
public:
	enum TYPE { MARKER_4X4, MARKER_5X5, MARKER_6X6, MARKER_7X7 };

	Marker(TYPE type, float size);

	virtual bool Detect(cv::Mat& image);
	
	cv::Vec3b Pixel(cv::Mat& image, cv::Point3f pos, int i = 0);

	bool LoadCameraParameters(const char* filepath);

protected:
	float _markerSize;
	std::vector<cv::Point3f> _corners;
	std::vector<cv::Point> _markerCorner;

	cv::Ptr<cv::aruco::Dictionary> _dictionary;
	cv::Ptr<cv::aruco::DetectorParameters> _detectorParams;
	cv::Mat _cameraMatrix, _distCoeffs;
	std::vector<int> _markerIdx;
	std::vector<cv::Vec3d> _rvecs, _tvecs;
};

