#pragma once
#ifdef WIN32
#pragma warning(disable:4819)
#endif // DEBUG
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>

#define AVERAGE_LENGTH 10

class MovingAvg
{
public:
	MovingAvg();

	/// <summary>
	/// Constructor
	/// </summary>
	/// <param name="tvec"></param>
	/// <param name="rvec"></param>
	MovingAvg(cv::Vec3d& tvec, cv::Vec3d& rvec);

	/// <summary>
	/// Update vectors
	/// </summary>
	/// <param name="tvec"></param>
	/// <param name="rvec"></param>
	void Update(cv::Vec3d& tvec, cv::Vec3d& rvec);

private:
	cv::Vec3d _tvecs[AVERAGE_LENGTH];
	cv::Vec3d _rvecs[AVERAGE_LENGTH];

	int _current;
};

