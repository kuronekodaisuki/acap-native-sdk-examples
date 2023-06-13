#pragma once
#include <ctime>
#include "iJSON.h"

class SnowDepth : public iJSON
{
public:
	SnowDepth(float depth = 0);

	bool Send(std::ostream& stream);

	float _depth;
	time_t _time;
};

