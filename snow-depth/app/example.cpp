/**
 * Copyright (C) 2021 Axis Communications AB, Lund, Sweden
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <syslog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include "larod.h"

#include "SnowMeasurement/SnowMeasurement.h"
#include "imgprovider.h"

using namespace cv;

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

int main(int argc, char* argv[])
{
  openlog("snow_depth", LOG_PID|LOG_CONS, LOG_USER);
  syslog(LOG_INFO, "Running OpenCV example with VDO as video source");
  ImgProvider_t* provider = NULL;

	// マーカーサイズ40センチ、ポール長さ24メートル
	SnowDetector detector(SnowDetector::MARKER_6X6, 0.4f, 2.75f);
  if (2 <= argc)
  {
    if (detector.LoadCameraParameters(argv[1]))
      syslog(LOG_INFO, "%s loaded", argv[1]);
    else
      syslog(LOG_ERR, "%s failed to load", argv[1]);
  }
  else
  {
    syslog(LOG_INFO, "No camera parameter file");
  }

  Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_100);
  std::vector<int> markerIds;
  std::vector<std::vector<Point2f>> markerCorners;

  // The desired width and height of the BGR frame
  unsigned int width = 1024;
  unsigned int height = 576;

  // chooseStreamResolution gets the least resource intensive stream
  // that exceeds or equals the desired resolution specified above
  unsigned int streamWidth = 0;
  unsigned int streamHeight = 0;
  if (!chooseStreamResolution(width, height, &streamWidth,
                              &streamHeight)) {
      syslog(LOG_ERR, "%s: Failed choosing stream resolution", __func__);
      exit(1);
  }

  syslog(LOG_INFO, "Creating VDO image provider and creating stream %d x %d",
          streamWidth, streamHeight);
  provider = createImgProvider(streamWidth, streamHeight, 2, VDO_FORMAT_YUV);
  if (!provider) {
    syslog(LOG_ERR, "%s: Failed to create ImgProvider", __func__);
    exit(2);
  }

  syslog(LOG_INFO, "Start fetching video frames from VDO");
  if (!startFrameFetch(provider)) {
    syslog(LOG_ERR, "%s: Failed to fetch frames from VDO", __func__);
    exit(3);
  }

  // Create the background subtractor
  Ptr<BackgroundSubtractorMOG2> bgsub = createBackgroundSubtractorMOG2();

  // Create the filtering element. Its size influences what is considered
  // noise, with a bigger size corresponding to more denoising
  Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));

  // Create OpenCV Mats for the camera frame (nv12), the converted frame (bgr)
  // and the foreground frame that is outputted by the background subtractor
  Mat bgr_mat = Mat(height, width, CV_8UC3);
  Mat nv12_mat = Mat(height * 3 / 2, width, CV_8UC1);
  Mat fg;

  while (true)
  {
    // Get the latest NV12 image frame from VDO using the imageprovider
    VdoBuffer* buf = getLastFrameBlocking(provider);
    if (!buf) {
      syslog(LOG_INFO, "No more frames available, exiting");
      exit(0);
    }

    // Assign the VDO image buffer to the nv12_mat OpenCV Mat.
    // This specific Mat is used as it is the one we created for NV12,
    // which has a different layout than e.g., BGR.
    nv12_mat.data = static_cast<uint8_t*>(vdo_buffer_get_data(buf));

    // Convert the NV12 data to BGR
    cvtColor(nv12_mat, bgr_mat, COLOR_YUV2BGR_NV12, 3);

    float depth = detector.Detect(bgr_mat);
    syslog(LOG_INFO, "'%d': %f", detector.GetStatus(), depth);


    aruco::detectMarkers(bgr_mat, dictionary, markerCorners, markerIds);
    aruco::drawDetectedMarkers(bgr_mat, markerCorners);
    if (0 < markerIds.size())
    {
      syslog(LOG_INFO, "%d Detect", markerIds[0]);
    }
    else
    {
      syslog(LOG_ERR, "Not Detected");
    }

    /*
    // Perform background subtraction on the bgr image with
    // learning rate 0.005. The resulting image should have
    // pixel intensities > 0 only where changes have occurred
    bgsub->apply(bgr_mat, fg, 0.005);

    // Filter noise from the image with the filtering element
    morphologyEx(fg, fg, MORPH_OPEN, kernel);

    // We define movement in the image as any pixel being non-zero
    int nonzero_pixels = countNonZero(fg);
    if (nonzero_pixels > 0) {
      syslog(LOG_INFO, "Motion detected: YES");
    } else {
      syslog(LOG_INFO, "Motion detected: NO");
    }
    */

    // Release the VDO frame buffer
    returnFrame(provider, buf);
  }
  return EXIT_SUCCESS;
}
