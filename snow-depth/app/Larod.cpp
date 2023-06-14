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

Larod::~Larod()
{
  if (_connection)
    larodDisconnect(&_connection, NULL);
  if (_model)
    larodDestroyModel(&_model);

  larodClearError(&_error);
}

bool Larod::LoadModel(const char* filename)
{
    // Create larod models
    syslog(LOG_INFO, "Create larod models");
    const int larodModelFd = open(filename, O_RDONLY);
    if (0 <= larodModelFd)
    {
        _model = larodLoadModel(_connection, larodModelFd, _device, LAROD_ACCESS_PRIVATE,
                                 "object_detection", NULL, &_error);
        close(larodModelFd);
        if (!_model)
        {
          syslog(LOG_ERR, "%s: Unable to load model: %s", __func__, _error->msg);
          return false;
        }
        else
          return true;
    }
    else
    {
        syslog(LOG_ERR, "Unable to open model file %s: %s", filename, strerror(errno));
        return false;
    }
}

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

bool createAndMapTmpFile(char* fileName, size_t fileSize, void** mappedAddr, int* convFd)
{
    syslog(LOG_INFO, "%s: Setting up a temp fd with pattern %s and size %zu", __func__,
           fileName, fileSize);

    int fd = mkstemp(fileName);
    if (0 <= fd)
    {
      // Allocate enough space in for the fd.
      if (0 <= ftruncate(fd, (off_t) fileSize))
      {
        // Remove since we don't actually care about writing to the file system.
        if (!unlink(fileName))
        {
          // Get an address to fd's memory for this process's memory space.
          void* data = mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

          if (data != MAP_FAILED)
          {
            // SUCCESS
            *mappedAddr = data;
            *convFd = fd;
            return true;
          }
          else
          {
              syslog(LOG_ERR, "%s: Unable to mmap temp file %s: %s", __func__, fileName, strerror(errno));
          }
        }
        else
        {
            syslog(LOG_ERR, "%s: Unable to unlink from temp file %s: %s", __func__, fileName, strerror(errno));
        }
      }
      else
      {
          syslog(LOG_ERR, "%s: Unable to truncate temp file %s: %s", __func__, fileName, strerror(errno));
      }
    }
    else
    {
        syslog(LOG_ERR, "%s: Unable to open temp file %s: %s", __func__, fileName, strerror(errno));
    }

    // ERROR
   if (fd >= 0) {
        close(fd);
    }
    return false;
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
