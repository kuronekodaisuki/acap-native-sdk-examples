#include <unistd.h>

class Map
{
public:
    Map(size_t size, char* whereToCreate);
    ~Map();
    int GetHandle() { return _handle;}
    void* GetPtr() {return _mappedAddr;}

private:
    size_t _size = 0;
    int _handle = -1;
    void* _mappedAddr = nullptr;
    char* _filename = nullptr;
};