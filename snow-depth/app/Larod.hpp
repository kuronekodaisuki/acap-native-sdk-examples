#include "larod.h"

class Larod
{
public:
    Larod(const char* chip = "cpu-tflite");
    ~Larod();

    bool LoadModel(const char* filename, const char* modelname = "inference");
    bool DoInference();

private:
    larodConnection* _connection;
    larodModel* _model;
    const larodDevice* _device;
    const larodJobRequest* _request;
    larodError* _error;
};