#include <vector>
#include "larod.h"

#include "Map.hpp"

class Larod
{
public:
    Larod(const char* chip = "cpu-tflite");
    ~Larod();

    bool LoadModel(const char* filename, size_t width, size_t height, size_t channels = 3, const char* modelname = "inference");

    bool DoInference();

private:
    larodConnection* _connection;
    larodModel* _model;
    const larodDevice* _device;
    const larodJobRequest* _request;
    larodTensor** _inputTensors;
    larodTensor** _outputTensors;
    size_t _numInputs;
    size_t _numOutputs;
    std::vector<Map> _inputs;
    std::vector<Map> _outputs;
    larodError* _error;
};