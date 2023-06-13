#pragma once
//
// iJSON -- JSON変換してストリームに出力
//
#include <iostream>

class iJSON
{
public:
	virtual bool Send(std::ostream& stream) = 0;
};