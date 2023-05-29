#include "MovingAvg.h"

MovingAvg::MovingAvg()
{
	_current = 0;
}

MovingAvg::MovingAvg(cv::Vec3d& tvec, cv::Vec3d& rvec)
{
	for (int i = 0; i < AVERAGE_LENGTH; i++)
	{
		_tvecs[i] = tvec;
		_rvecs[i] = rvec;
	}
	_current = 0;
}

void MovingAvg::Update(cv::Vec3d& tvec, cv::Vec3d& rvec)
{
	_tvecs[_current] = tvec;
	_rvecs[_current] = rvec;
	
	_current++;
	if (_current == AVERAGE_LENGTH)
		_current = 0;
	tvec = _tvecs[0];
	rvec = _rvecs[0];
	for (int i = 1; i < AVERAGE_LENGTH; i++)
	{
		tvec += _tvecs[i];
		rvec += _rvecs[i];
	}
	tvec /= AVERAGE_LENGTH;
	rvec /= AVERAGE_LENGTH;
}