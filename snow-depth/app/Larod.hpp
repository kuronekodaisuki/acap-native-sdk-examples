#include "larod.h"

class Larod
{
public:
    Larod(const char* chip);
    ~Larod();

    bool LoadModel(const char* filename);
private:
    larodConnection* _connection;
    const larodDevice* _device;
    larodModel* _model;
    larodError* _error;
};