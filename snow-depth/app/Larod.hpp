#include <vector>
#include "larod.h"
#include "vdo-stream.h"

#include "Map.hpp"
#include "Labels.h"

class Larod
{
public:
    /// @brief Constructor
    /// @param streamWidth
    /// @param streamHeight
    /// @param device cpu-tflite | google-edge-tpu-tflite | axis-a7-gpu-tflite
    Larod(size_t streamWidth, size_t streamHeight, const char* device = "cpu-tflite");

    /// @brief Destructor
    ~Larod();

    /// @brief Load labels
    /// @param filename
    /// @return
    size_t LoadLabels(const char* filename);

    /// @brief Enumerate device
    void EnumerateDevices();

    /// @brief Load inference model
    /// @param filename
    /// @param width
    /// @param height
    /// @param channels
    /// @param modelname
    /// @return
    bool LoadModel(const char* filename, size_t width, size_t height, size_t channels = 3, const char* modelname = "inference");

    /// @brief Do inference
    /// @return
    virtual bool DoInference(VdoBuffer* yuvData);

protected:
    virtual bool CreatePreProcessModel();
    virtual bool PostProcess();

    const char* _chip;
    size_t _channels;
    size_t _streamWidth;
    size_t _streamHeight;
    size_t _modelWidth;
    size_t _modelHeight;
    size_t _yuyvBufferSize;
    float _threshold = 0.5;
    Labels _labels;
    const larodDevice* _device;
    const larodJobRequest* _InferRequest;
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