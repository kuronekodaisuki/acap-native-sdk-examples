
#include "SnowDepth.h"

/// <summary>
/// Constructor
/// </summary>
/// <param name="depth"></param>
SnowDepth::SnowDepth(float depth)
{
	_depth = depth;
	_time = time(0);
}

/// <summary>
/// JSONèoóÕÇÃé¿ëï
/// </summary>
/// <param name="stream"></param>
/// <returns></returns>
bool SnowDepth::Send(std::ostream& stream)
{
	static char datetime[78];
	static char buffer[120];
	strftime(datetime, sizeof(datetime) - 1, "%Y-%m-%d %H:%M:%S", localtime(&_time));
	sprintf(buffer, "{\"depth\":%f, \"time\":\"%s\"}", _depth, datetime);
	stream << buffer;
	return true;
}
