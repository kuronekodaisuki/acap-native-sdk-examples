#pragma once
#include "Marker.h"


class Flood : public Marker
{
public:
	Flood(TYPE type, float size);

	bool Detect(cv::Mat& image);

	void Scan(cv::Mat& image);

private:
	std::vector<cv::Point3f> _scanArea;

};

