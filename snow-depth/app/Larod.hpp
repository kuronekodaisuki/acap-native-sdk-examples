#include <vector>
#include "larod.h"

#include "Map.hpp"

class Larod
{
public:
    Larod(const char* chip = "cpu-tflite");
    ~Larod();

    bool LoadModel(const char* filename, size_t width, size_t height, size_t channels = 3, const char* modelname = "inference");

    bool PreProcessModel(size_t streamWidth, size_t streamHeight);

    bool DoInference();

private:
    const char* _chip;
    size_t _modelWidth;
    size_t _modelHeight;
    const larodDevice* _device;
    const larodJobRequest* _request;
    const larodJobRequest* _ppRequest;
    larodConnection* _connection;
    larodMap* _ppMap;
    larodMap* _cropMap;
    larodModel* _model;
    larodModel* _ppModel;
    larodTensor** _inputTensors;
    larodTensor** _outputTensors;
    larodTensor** _ppInputTensors;
    larodTensor** _ppOutputTensors;
    size_t _numInputs;
    size_t _numOutputs;
    //Map _preProcess;
    //Map _crop;
    std::vector<Map> _inputs;
    std::vector<Map> _outputs;
    larodError* _error;
};