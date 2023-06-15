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
//char CONV_PP_FILE_PATTERN[] = "/tmp/larod.pp.test-XXXXXX";
//char CROP_FILE_PATTERN[] = "/tmp/crop.test-XXXXXX";

bool createAndMapTmpFile(char* fileName, size_t fileSize, void** mappedAddr, int* convFd);

/// @brief Constructor
/// @param chip
Larod::Larod(const char* chip)
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
