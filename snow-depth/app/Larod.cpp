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
    }
    else
    {
        syslog(LOG_ERR, "%s: Could not connect to larod: %s", __func__, _error->msg);
    }
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