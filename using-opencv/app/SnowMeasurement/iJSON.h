#pragma once
//
// iJSON -- JSON�ϊ����ăX�g���[���ɏo��
//
#include <iostream>

class iJSON
{
public:
	virtual bool Send(std::ostream& stream) = 0;
};