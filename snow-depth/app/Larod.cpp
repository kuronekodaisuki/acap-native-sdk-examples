#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <syslog.h>
#include <unistd.h>
#include <string.h>

#include "Larod.hpp"

// Hardcode to use three image "color" channels (eg. RGB).
const size_t CHANNELS = 3;
// Hardcode to set output bytes of four tensors from MobileNet V2 model.
const size_t FLOATSIZE = 4;
const size_t TENSOR1SIZE = 80 * FLOATSIZE;
const size_t TENSOR2SIZE = 20 * FLOATSIZE;
const size_t TENSOR3SIZE = 20 * FLOATSIZE;
const size_t TENSOR4SIZE = 1 * FLOATSIZE;

// Name patterns for the temp file we will create.
char CONV_INP_FILE_PATTERN[] = "/tmp/larod.in.test-XXXXXX";
char CONV_OUT1_FILE_PATTERN[] = "/tmp/larod.out1.test-XXXXXX";
char CONV_OUT2_FILE_PATTERN[] = "/tmp/larod.out2.test-XXXXXX";
char CONV_OUT3_FILE_PATTERN[] = "/tmp/larod.out3.test-XXXXXX";
char CONV_OUT4_FILE_PATTERN[] = "/tmp/larod.out4.test-XXXXXX";
char CONV_PP_FILE_PATTERN[] = "/tmp/larod.pp.test-XXXXXX";
char CROP_FILE_PATTERN[] = "/tmp/crop.test-XXXXXX";

/// @brief Constructor
/// @param chip
Larod::Larod(const char* chip): _chip(chip)
{
    // Set up larod connection.
    if (larodConnect(&_connection, &_error))
    {
      _device = larodGetDevice(_connection, chip, 0, &_error);
      if (_device)
        syslog(LOG_INFO, "%s device connected", chip);
      else
      {
        syslog(LOG_ERR, "Can't connect device %s", _error->msg);
        // List available chip id:s
        size_t numDevices = 0;
        syslog(LOG_INFO, "Available chip IDs:");
        const larodDevice** devices;
        devices = larodListDevices(_connection, &numDevices, &_error);
        for (size_t i = 0; i < numDevices; ++i)
        {
          syslog(LOG_INFO, "%s: %s", "Chip", larodGetDeviceName(devices[i], &_error));;
        }
      }
    }
    else
    {
        syslog(LOG_ERR, "%s: Could not connect to larod: %s", __func__, _error->msg);
    }
}

/// @brief Destructor
Larod::~Larod()
{
  if (_model)
  {
    delete _preProcess;
    delete _crop;
    larodDestroyTensors(_connection, &_inputTensors, _numInputs, &_error);
    larodDestroyTensors(_connection, &_outputTensors, _numOutputs, &_error);
    larodDestroyModel(&_model);
    for (size_t i = 0; i < _numInputs; i++)
    {
      _inputs.pop_back();
    }
    for (size_t i = 0; i < _numOutputs; i++)
    {
      _outputs.pop_back();
    }
  }
  if (_connection)
  {
    larodDisconnect(&_connection, NULL);
  }

  larodClearError(&_error);
}

/// @brief Load model and get Inputs and Outputs Tensors
/// @param filename
/// @param width
/// @param height
/// @param channels
/// @param modelname
/// @return
bool Larod::LoadModel(const char* filename, size_t width, size_t height, size_t channels, const char* modelname)
{
  _modelWidth = width;
  _modelHeight = height;
    // Create larod models
    syslog(LOG_INFO, "Create larod models");
    const int larodModelFd = open(filename, O_RDONLY);
    if (0 <= larodModelFd)
    {
        _model = larodLoadModel(_connection, larodModelFd, _device, LAROD_ACCESS_PRIVATE,
                                 modelname, NULL, &_error);
        close(larodModelFd);
        if (!_model)
        {
          syslog(LOG_ERR, "%s: Unable to load model: %s", __func__, _error->msg);
          return false;
        }
        else
        {
          syslog(LOG_INFO, "Model %s loaded", filename);

          _inputTensors = larodCreateModelInputs(_model, &_numInputs, &_error);
          _outputTensors = larodCreateModelOutputs(_model, &_numOutputs, &_error);
          syslog(LOG_INFO, "%d inputs, %d outputs", _numInputs, _numOutputs);

          _inputs.push_back(Map(width * height * channels, CONV_INP_FILE_PATTERN));

          for (size_t i = 0; i < _numOutputs; i++)
          {
            switch (i)
            {
              case 0:
                _outputs.push_back(Map(TENSOR1SIZE, CONV_OUT1_FILE_PATTERN));
                break;
              case 1:
                _outputs.push_back(Map(TENSOR2SIZE, CONV_OUT2_FILE_PATTERN));
                break;
              case 2:
                _outputs.push_back(Map(TENSOR3SIZE, CONV_OUT3_FILE_PATTERN));
                break;
              case 3:
                _outputs.push_back(Map(TENSOR4SIZE, CONV_OUT4_FILE_PATTERN));
                break;
            }
          }

          _request = larodCreateJobRequest(_model,
                                          _inputTensors, _numInputs,
                                          _outputTensors, _numOutputs,
                                          NULL, &_error);
          if (_request)
          {
            for (size_t i = 0; i < _numInputs; i++)
            {
              //createAndMapTmpFile(CONV_INP_FILE_PATTERN, width * height * channels, )
            }
          }
          return true;
        }
    }
    else
    {
        syslog(LOG_ERR, "Unable to open model file %s: %s", filename, strerror(errno));
        return false;
    }
}

bool Larod::PreProcessModel(size_t streamWidth, size_t streamHeight)
{
    // Calculate crop image
    // 1. The crop area shall fill the input image either horizontally or
    //    vertically.
    // 2. The crop area shall have the same aspect ratio as the output image.
    syslog(LOG_INFO, "Calculate crop image");
    float destWHratio = (float) _modelWidth / (float) _modelHeight;
    float cropW = (float) streamWidth;
    float cropH = cropW / destWHratio;
    if (cropH > (float) streamHeight) {
        cropH = (float) streamHeight;
        cropW = cropH * destWHratio;
    }
    unsigned int clipW = (unsigned int)cropW;
    unsigned int clipH = (unsigned int)cropH;
    unsigned int clipX = (streamWidth - clipW) / 2;
    unsigned int clipY = (streamHeight - clipH) / 2;
    syslog(LOG_INFO, "Crop VDO image X=%d Y=%d (%d x %d)", clipX, clipY, clipW, clipH);

    // Create preprocessing maps
    syslog(LOG_INFO, "Create preprocessing maps");
    _ppMap = larodCreateMap(&_error);
    if (!_ppMap) {
        syslog(LOG_ERR, "Could not create preprocessing larodMap %s", _error->msg);
        return false;
    }
    if (!larodMapSetStr(_ppMap, "image.input.format", "nv12", &_error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", _error->msg);
        return false;
    }
    if (!larodMapSetIntArr2(_ppMap, "image.input.size", streamWidth, streamHeight, &_error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", _error->msg);
        return false;
    }
    if(_chip != "ambarella-cvflow"){
        if (!larodMapSetStr(_ppMap, "image.output.format", "rgb-interleaved", &_error)) {
            syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", _error->msg);
        return false;
        }
    } else {
        if (!larodMapSetStr(_ppMap, "image.output.format", "rgb-planar", &_error)) {
            syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", _error->msg);
        return false;
        }
    }
    if (!larodMapSetIntArr2(_ppMap, "image.output.size", _modelWidth, _modelHeight, &_error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", _error->msg);
        return false;
    }
    _cropMap = larodCreateMap(&_error);
    if (!_cropMap) {
        syslog(LOG_ERR, "Could not create preprocessing crop larodMap %s", _error->msg);
        return false;
    }
    if (!larodMapSetIntArr4(_cropMap, "image.input.crop", clipX, clipY, clipW, clipH, &_error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", _error->msg);
        return false;
    }

    // Use libyuv as image preprocessing backend
    const char* larodLibyuvPP = "cpu-proc";
    const larodDevice* dev_pp = larodGetDevice(_connection, larodLibyuvPP, 0, &_error);
    _ppModel = larodLoadModel(_connection, -1, dev_pp, LAROD_ACCESS_PRIVATE, "", _ppMap, &_error);
    if (!_ppModel) {
            syslog(LOG_ERR, "Unable to load preprocessing model with chip %s: %s", larodLibyuvPP, _error->msg);
        return false;
    } else {
           syslog(LOG_INFO, "Loading preprocessing model with chip %s", larodLibyuvPP);
    }

    // Create input/output tensors
    size_t ppNumInputs;
    size_t ppNumOutputs;
    syslog(LOG_INFO, "Create input/output tensors");
    _ppInputTensors = larodCreateModelInputs(_ppModel, &ppNumInputs, &_error);
    if (!_ppInputTensors) {
        syslog(LOG_ERR, "Failed retrieving input tensors: %s", _error->msg);
        return false;
    }
    _ppOutputTensors = larodCreateModelOutputs(_ppModel, &ppNumOutputs, &_error);
    if (!_ppOutputTensors) {
        syslog(LOG_ERR, "Failed retrieving output tensors: %s", _error->msg);
        return false;
    }

    // Determine tensor buffer sizes
    syslog(LOG_INFO, "Determine tensor buffer sizes");
    const larodTensorPitches* ppInputPitches = larodGetTensorPitches(_ppInputTensors[0], &_error);
    if (!ppInputPitches) {
        syslog(LOG_ERR, "Could not get pitches of tensor: %s", _error->msg);
        return false;
    }
    size_t yuyvBufferSize = ppInputPitches->pitches[0];
    const larodTensorPitches* outputPitches = larodGetTensorPitches(_outputTensors[0], &_error);
    if (!outputPitches) {
        syslog(LOG_ERR, "Could not get pitches of tensor: %s", _error->msg);
        return false;
    }
    //outputBufferSize = outputPitches->pitches[0];

    _preProcess = new Map(yuyvBufferSize, CONV_PP_FILE_PATTERN);
    //_crop = new Map(rawWidth * rawHeight * CHANNELS, CROP_FILE_PATTERN);

/*
    // Connect tensors to file descriptors
    syslog(LOG_INFO, "Connect tensors to file descriptors");
    if (!larodSetTensorFd(_ppInputTensors[0], ppInputFd, &_error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", _error->msg);
        return false;
    }
    if (!larodSetTensorFd(_ppOutputTensors[0], larodInputFd, &_error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", _error->msg);
        return false;
    }
*/

    // Create job requests
    syslog(LOG_INFO, "Create job requests");
    _ppRequest = larodCreateJobRequest(_ppModel, _ppInputTensors, ppNumInputs,
        _ppOutputTensors, ppNumOutputs, _cropMap, &_error);
    if (!_ppRequest) {
        syslog(LOG_ERR, "Failed creating preprocessing job request: %s", _error->msg);
        return false;
    }
    return true;

}

/// @brief Do Inference
/// @return
bool Larod::DoInference()
{
  if (_connection && _request)
  {
    return larodRunJob(_connection, _request, &_error);
  }
  else
  {
    syslog(LOG_ERR, "Failed to Inference");
    return false;
  }
}


larodMap* CreatePreProcessMap(unsigned int streamWidth, unsigned int streamHeight, unsigned int inputWidth, unsigned int inputHeight)
{
  larodError* error = NULL;
  larodMap* ppMap = larodCreateMap(&error);
    if (!ppMap) {
        syslog(LOG_ERR, "Could not create preprocessing larodMap %s", error->msg);
        return nullptr;
    }
    if (!larodMapSetStr(ppMap, "image.input.format", "nv12", &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        return nullptr;
    }
    if (!larodMapSetIntArr2(ppMap, "image.input.size", streamWidth, streamHeight, &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        return nullptr;
    }
    if (!larodMapSetStr(ppMap, "image.output.format", "rgb-planar", &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
    return nullptr;
    }
    if (!larodMapSetIntArr2(ppMap, "image.output.size", inputWidth, inputHeight, &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        return nullptr;
    }
    return ppMap;
}

larodMap* CreateCropMap(unsigned int clipX, unsigned int clipY, unsigned int clipW, unsigned int clipH)
{
  larodError* error = NULL;
  larodMap* cropMap = larodCreateMap(&error);
    if (!cropMap) {
        syslog(LOG_ERR, "Could not create preprocessing crop larodMap %s", error->msg);
        return nullptr;
    }
    if (!larodMapSetIntArr4(cropMap, "image.input.crop", clipX, clipY, clipW, clipH, &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        return nullptr;
    }
    return cropMap;
}
