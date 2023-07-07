#pragma once
#include "Marker.h"


class Flood : public Marker
{
public:
	Flood(float size = 0.15f, TYPE type = TYPE::MARKER_6X6);

	bool Detect(cv::Mat& image);

	void Scan(cv::Mat& image);

private:
	std::vector<cv::Point3f> _scanArea;

};

