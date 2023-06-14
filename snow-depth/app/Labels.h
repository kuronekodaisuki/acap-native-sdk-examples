#pragma once

#include <vector>
#include <string>

#ifdef WIN32
#ifdef EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif
#include <io.h>
#define syslog()
#pragma warning(disable: 4996)
#else
#define API
#include <unistd.h>
#include <syslog.h>
#include <string.h>
#endif

class API Labels
{
public:
	//Labels();
	//~Labels();

	size_t Load(const char* filename);
	size_t GetCount() { return _labels.size(); }
	const char* operator[](unsigned int index);

private:
	std::vector<std::string> _labels;
};

