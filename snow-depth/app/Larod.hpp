#include <vector>
#include "larod.h"

#include "Map.hpp"
#include "Labels.h"

class Larod
{
public:
    Larod(size_t streamWidth, size_t streamHeight, const char* chip = "cpu-tflite");
    ~Larod();

    size_t LoadLabels(const char* filename);

    void EnumerateDevices();

    bool LoadModel(const char* filename, size_t width, size_t height, size_t channels = 3, const char* modelname = "inference");

    virtual bool PreProcessModel();

    virtual bool DoInference();

    virtual bool PostProcess();
private:
    const char* _chip;
    size_t _channels;
    size_t _streamWidth;
    size_t _streamHeight;
    size_t _modelWidth;
    size_t _modelHeight;
    float _threshold = 0.5;
    Labels _labels;
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
    Map* _preProcess = nullptr;
    Map* _crop = nullptr;
    std::vector<Map> _inputs;
    std::vector<Map> _outputs;
    larodError* _error;
};