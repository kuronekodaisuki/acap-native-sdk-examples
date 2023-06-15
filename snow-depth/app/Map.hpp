#include <unistd.h>

class Map
{
public:
    Map(size_t size, char* whereToCreate);
    ~Map();

private:
    size_t _size = 0;
    int _handle = -1;
    void* _mappedAddr = nullptr;
};