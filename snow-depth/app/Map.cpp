#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <syslog.h>
#include <string.h>

#include "Map.hpp"

Map::Map(size_t fileSize, char* fileName): _size(fileSize), _filename(fileName)
{
    syslog(LOG_INFO, "%s", __func__);

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
            _mappedAddr = data;
            _handle = fd;
            syslog(LOG_INFO, "Mapped with pattern %s and size %zu as handle:%d addr:%x", fileName, fileSize, _handle, _mappedAddr);
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
}

Map::~Map()
{
  syslog(LOG_INFO, "Map deleted size:%d file:%s", _size, _filename);
  if (_mappedAddr != nullptr)
    munmap(_mappedAddr, _size);
  if (0 <= _handle)
    close(_handle);
  _handle = -1;
  _mappedAddr = nullptr;
}