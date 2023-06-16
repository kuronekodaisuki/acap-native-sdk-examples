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
Larod::Larod(size_t streamWidth, size_t streamHeight, const char* chip):
_streamWidth(streamWidth), _streamHeight(streamHeight), _chip(chip)
{
    // Set up larod connection.
    if (larodConnect(&_connection, &_error))
    {
      _device = larodGetDevice(_connection, _chip, 0, &_error);
      if (_device)
        syslog(LOG_INFO, "%s device connected", _chip);
      else
      {
        syslog(LOG_ERR, "Can't connect device %s", _error->msg);
        EnumerateDevices();
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
  syslog(LOG_INFO, "Destroy Larod object");
  larodClearError(&_error);
}

/// @brief Enumerate devices
void Larod::EnumerateDevices()
{
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

size_t Larod::LoadLabels(const char* filename)
{
  return _labels.Load(filename);
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
  _channels = channels;

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

          // Prepare input and output buffer
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

          // Create Request of inference
          _InferRequest = larodCreateJobRequest(_model,
                                          _inputTensors, _numInputs,
                                          _outputTensors, _numOutputs,
                                          NULL, &_error);
          return CreatePreProcessModel();
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
bool Larod::DoInference(VdoBuffer* buf)
{
  if (_connection && _InferRequest)
  {
    struct timeval startTs, endTs;
    unsigned int elapsedMs = 0;

        // Get data from latest frame.
        uint8_t* nv12Data = (uint8_t*) vdo_buffer_get_data(buf);

        // Covert image data from NV12 format to interleaved uint8_t RGB format.
        gettimeofday(&startTs, NULL);

        // Convert YUV to RGB
        memcpy(_preProcess->GetPtr(), nv12Data, _yuyvBufferSize);
        if (!larodRunJob(_connection, _ppRequest, &_error)) {
            syslog(LOG_ERR, "Unable to run job to preprocess model: %s (%d)",
                   _error->msg, _error->code);
            return false;
        }

        gettimeofday(&endTs, NULL);

        elapsedMs = (unsigned int) (((endTs.tv_sec - startTs.tv_sec) * 1000) +
                                    ((endTs.tv_usec - startTs.tv_usec) / 1000));
        syslog(LOG_INFO, "Converted image in %u ms", elapsedMs);

    // Since larodOutputAddr points to the beginning of the fd we should
    // rewind the file position before each job.
    if (lseek(_outputs[0].GetHandle(), 0, SEEK_SET) == -1) {
        syslog(LOG_ERR, "Unable to rewind output file position: %s",
                strerror(errno));
      return false;
    }

    if (lseek(_outputs[1].GetHandle(), 0, SEEK_SET) == -1) {
        syslog(LOG_ERR, "Unable to rewind output file position: %s",
                strerror(errno));
      return false;
    }

    if (lseek(_outputs[2].GetHandle(), 0, SEEK_SET) == -1) {
        syslog(LOG_ERR, "Unable to rewind output file position: %s",
                strerror(errno));
      return false;
    }

    if (lseek(_outputs[3].GetHandle(), 0, SEEK_SET) == -1) {
        syslog(LOG_ERR, "Unable to rewind output file position: %s",
                strerror(errno));
      return false;
    }

    gettimeofday(&startTs, NULL);
    if (!larodRunJob(_connection, _InferRequest, &_error)) {
        syslog(LOG_ERR, "Unable to run inference on model %s (%d)",
                _error->msg, _error->code);
      return false;
    }
    gettimeofday(&endTs, NULL);

    elapsedMs = (unsigned int) (((endTs.tv_sec - startTs.tv_sec) * 1000) +
                                ((endTs.tv_usec - startTs.tv_usec) / 1000));
    syslog(LOG_INFO, "Ran inference for %u ms", elapsedMs);

    return PostProcess();
  }
  syslog(LOG_ERR, "Failed to Inference");
  return false;
}

/// @brief Create pre process model
/// @return
bool Larod::CreatePreProcessModel()
{
    // Calculate crop image
    // 1. The crop area shall fill the input image either horizontally or
    //    vertically.
    // 2. The crop area shall have the same aspect ratio as the output image.
    syslog(LOG_INFO, "Calculate crop image");
    float destWHratio = (float) _modelWidth / (float) _modelHeight;
    float cropW = (float) _streamWidth;
    float cropH = cropW / destWHratio;
    if (cropH > (float) _streamHeight) {
        cropH = (float) _streamHeight;
        cropW = cropH * destWHratio;
    }
    unsigned int clipW = (unsigned int)cropW;
    unsigned int clipH = (unsigned int)cropH;
    unsigned int clipX = (_streamWidth - clipW) / 2;
    unsigned int clipY = (_streamHeight - clipH) / 2;
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
    if (!larodMapSetIntArr2(_ppMap, "image.input.size", _streamWidth, _streamHeight, &_error)) {
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
    _yuyvBufferSize = ppInputPitches->pitches[0];
    const larodTensorPitches* outputPitches = larodGetTensorPitches(_outputTensors[0], &_error);
    if (!outputPitches) {
        syslog(LOG_ERR, "Could not get pitches of tensor: %s", _error->msg);
        return false;
    }
    //outputBufferSize = outputPitches->pitches[0];

    _preProcess = new Map(_yuyvBufferSize, CONV_PP_FILE_PATTERN);
    _crop = new Map(_streamWidth * _streamHeight * _channels, CROP_FILE_PATTERN);

    // Connect tensors to file descriptors
    syslog(LOG_INFO, "Connect tensors to file descriptors");
    if (!larodSetTensorFd(_ppInputTensors[0], _preProcess->GetHandle(), &_error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", _error->msg);
        return false;
    }
    if (!larodSetTensorFd(_ppOutputTensors[0], _inputs[0].GetHandle(), &_error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", _error->msg);
        return false;
    }

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

/// @brief Post process
/// @return
bool Larod::PostProcess()
{
  float* locations = (float*) _outputs[0].GetPtr();
  float* classes = (float*) _outputs[1].GetPtr();
  float* scores = (float*) _outputs[2].GetPtr();
  float* numberofdetections = (float*) _outputs[3].GetPtr();

  if ((int) numberofdetections[0] == 0)
  {
      syslog(LOG_INFO,"No object is detected");
  }
  else
  {
      for (int i = 0; i < numberofdetections[0]; i++)
      {
          float top = locations[4*i];
          float left = locations[4*i+1];
          float bottom = locations[4*i+2];
          float right = locations[4*i+3];

          unsigned int crop_x = left * _streamWidth;
          unsigned int crop_y = top * _streamHeight;
          unsigned int crop_w = (right - left) * _streamWidth;
          unsigned int crop_h = (bottom - top) * _streamHeight;

          if (scores[i] >= _threshold)
          {
              syslog(LOG_INFO, "Object %d: Classes: %s - Scores: %f - Locations: [%f,%f,%f,%f]",
                  i, _labels[(int) classes[i]], scores[i], top, left, bottom, right);
/*
              unsigned char* crop_buffer = crop_interleaved(cropAddr, rawWidth, rawHeight, CHANNELS,
                                                            crop_x, crop_y, crop_w, crop_h);

              unsigned long jpeg_size = 0;
              unsigned char* jpeg_buffer = NULL;
              struct jpeg_compress_struct jpeg_conf;
              set_jpeg_configuration(crop_w, crop_h, CHANNELS, args.quality, &jpeg_conf);
              buffer_to_jpeg(crop_buffer, &jpeg_conf, &jpeg_size, &jpeg_buffer);
              char file_name[32];
              snprintf(file_name, sizeof(char) * 32, "/tmp/detection_%i.jpg", i);
              jpeg_to_file(file_name, jpeg_buffer, jpeg_size);
              free(crop_buffer);
              free(jpeg_buffer);
*/
          }
      }
  }

  return true;
}