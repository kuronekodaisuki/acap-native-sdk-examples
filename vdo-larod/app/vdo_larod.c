/**
 * Copyright (C) 2021, Axis Communications AB, Lund, Sweden
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
#include <math.h>
#include <string.h>
#include <glib.h>
#include <cairo/cairo.h>
#include <axoverlay.h>
#include <syslog.h>

#include "imgprovider.h"
#include "larod.h"
#include "utility-functions.h"
#include "vdo-frame.h"
#include "vdo-types.h"

#define PALETTE_VALUE_RANGE 255.0
static gint overlay_id      = -1;
static gint overlay_id_text = -1;
static gint counter = 10;
static gint top_color = 1;
static gint bottom_color = 3;

volatile sig_atomic_t stopRunning = false;

/***** Drawing functions *****************************************************/

/**
 * brief Converts palette color index to cairo color value.
 *
 * This function converts the palette index, which has been initialized by
 * function axoverlay_set_palette_color to a value that can be used by
 * function cairo_set_source_rgba.
 *
 * param color_index Index in the palette setup.
 *
 * return color value.
 */
static gdouble
index2cairo(gint color_index)
{
  return ((color_index << 4) + color_index) / PALETTE_VALUE_RANGE;
}

/**
 * brief Draw a rectangle using palette.
 *
 * This function draws a rectangle with lines from coordinates
 * left, top, right and bottom with a palette color index and
 * line width.
 *
 * param context Cairo rendering context.
 * param left Left coordinate (x1).
 * param top Top coordinate (y1).
 * param right Right coordinate (x2).
 * param bottom Bottom coordinate (y2).
 * param color_index Palette color index.
 * param line_width Rectange line width.
 */
static void
draw_rectangle(cairo_t *context, gint left, gint top,
               gint right, gint bottom,
               gint color_index, gint line_width)
{
  gdouble val = 0;

  val = index2cairo(color_index);
  cairo_set_source_rgba(context, val, val, val, val);
  cairo_set_operator(context, CAIRO_OPERATOR_SOURCE);
  cairo_set_line_width(context, line_width);
  cairo_rectangle(context, left, top, right - left, bottom - top);
  cairo_stroke(context);
}

/**
 * brief Draw a text using cairo.
 *
 * This function draws a text with a specified middle position,
 * which will be adjusted depending on the text length.
 *
 * param context Cairo rendering context.
 * param pos_x Center position coordinate (x).
 * param pos_y Center position coordinate (y).
 */
static void
draw_text(cairo_t *context, gint pos_x, gint pos_y)
{
  cairo_text_extents_t te;
  cairo_text_extents_t te_length;
  gchar *str = NULL;
  gchar *str_length = NULL;

  //  Show text in black
  cairo_set_source_rgb(context, 0, 0, 0);
  cairo_select_font_face(context, "serif", CAIRO_FONT_SLANT_NORMAL,
                         CAIRO_FONT_WEIGHT_BOLD);
  cairo_set_font_size(context, 32.0);

  // Position the text at a fix centered position
  str_length = g_strdup_printf("Countdown  ");
  cairo_text_extents(context, str_length, &te_length);
  cairo_move_to(context, pos_x - te_length.width / 2, pos_y);
  g_free(str_length);

  // Add the counter number to the shown text
  str = g_strdup_printf("Countdown %i", counter);
  cairo_text_extents(context, str, &te);
  cairo_show_text(context, str);
  g_free(str);
}

/**
 * brief Setup an overlay_data struct.
 *
 * This function initialize and setup an overlay_data
 * struct with default values.
 *
 * param data The overlay data struct to initialize.
 */
static void
setup_axoverlay_data(struct axoverlay_overlay_data *data)
{
  axoverlay_init_overlay_data(data);
  data->postype = AXOVERLAY_CUSTOM_NORMALIZED;
  data->anchor_point = AXOVERLAY_ANCHOR_CENTER;
  data->x = 0.0;
  data->y = 0.0;
  data->scale_to_stream = FALSE;
}

/**
 * brief Setup palette color table.
 *
 * This function initialize and setup an palette index
 * representing ARGB values.
 *
 * param color_index Palette color index.
 * param r R (red) value.
 * param g G (green) value.
 * param b B (blue) value.
 * param a A (alpha) value.
 *
 * return result as boolean
 */
static gboolean
setup_palette_color(gint index, gint r, gint g, gint b, gint a)
{
  GError *error = NULL;
  struct axoverlay_palette_color color;

  color.red = r;
  color.green = g;
  color.blue = b;
  color.alpha = a;
  color.pixelate = FALSE;
  axoverlay_set_palette_color(index, &color, &error);
  if (error != NULL) {
    g_error_free(error);
    return FALSE;
  }

  return TRUE;
}

/**
 * brief Invoked on SIGINT. Makes app exit cleanly asap if invoked once, but
 * forces an immediate exit without clean up if invoked at least twice.
 *
 * param sig What signal has been sent.
 */
static void sigintHandler(int sig) {
    if (stopRunning) {
        syslog(LOG_INFO, "Interrupted again, exiting immediately without clean up.");

        signal(sig, SIG_DFL);
        raise(sig);

        return;
    }

    syslog(LOG_INFO, "Interrupted, starting graceful termination of app. Another "
           "interrupt signal will cause a forced exit.");

    // Tell the main thread to stop running inferences asap.
    stopRunning = true;
}

/**
 * brief Sets up and configures a connection to larod, and loads a model.
 *
 * Opens a connection to larod, which is tied to larodConn. After opening a
 * larod connection the chip specified by larodChip is set for the
 * connection. Then the model file specified by larodModelFd is loaded to the
 * chip, and a corresponding larodModel object is tied to model.
 *
 * param larodChip Specifier for which larod chip to use.
 * param larodModelFd Fd for a model file to load.
 * param larodConn Pointer to a larod connection to be opened.
 * param model Pointer to a larodModel to be obtained.
 * return False if error has occurred, otherwise true.
 */
static bool setupLarod(const char* chipString, const int larodModelFd,
                       larodConnection** larodConn, larodModel** model) {
    larodError* error = NULL;
    larodConnection* conn = NULL;
    larodModel* loadedModel = NULL;
    bool ret = false;

    // Set up larod connection.
    if (!larodConnect(&conn, &error)) {
        syslog(LOG_ERR, "%s: Could not connect to larod: %s", __func__, error->msg);
        goto end;
    }

    // List available chip id:s
    size_t numDevices = 0;
    syslog(LOG_INFO, "Available chip IDs:");
    const larodDevice** devices; 
    devices = larodListDevices(conn, &numDevices, &error);
    for (size_t i = 0; i < numDevices; ++i) {
            syslog(LOG_INFO, "%s: %s", "Chip", larodGetDeviceName(devices[i], &error));;
        }
    const larodDevice* dev = larodGetDevice(conn, chipString, 0, &error);
    loadedModel = larodLoadModel(conn, larodModelFd, dev, LAROD_ACCESS_PRIVATE,
                                 "Vdo Example App Model", NULL, &error);
    if (!loadedModel) {
        syslog(LOG_ERR, "%s: Unable to load model: %s", __func__, error->msg);
        goto error;
    }
    *larodConn = conn;
    *model = loadedModel;

    ret = true;

    goto end;

error:
    if (conn) {
        larodDisconnect(&conn, NULL);
    }

end:
    if (error) {
        larodClearError(&error);
    }

    return ret;
}

int SetupOverlay()
{
    GError *error = NULL;
    gint camera_height = 0;
    gint camera_width = 0;

    if(!axoverlay_is_backend_supported(AXOVERLAY_CAIRO_IMAGE_BACKEND)) {
        syslog(LOG_ERR, "AXOVERLAY_CAIRO_IMAGE_BACKEND is not supported");
        return 1;
    }

    //  Initialize the library
    struct axoverlay_settings settings;
    axoverlay_init_axoverlay_settings(&settings);
    //settings.render_callback = render_overlay_cb;
    //settings.adjustment_callback = adjustment_cb;
    settings.select_callback = NULL;
    settings.backend = AXOVERLAY_CAIRO_IMAGE_BACKEND;
    axoverlay_init(&settings, &error);
    if (error != NULL) {
        syslog(LOG_ERR, "Failed to initialize axoverlay: %s", error->message);
        g_error_free(error);
        return 1;
    }

    //  Setup colors
    if (!setup_palette_color(0, 0, 0, 0, 0) ||
        !setup_palette_color(1, 255, 0, 0, 255) ||
        !setup_palette_color(2, 0, 255, 0, 255) ||
        !setup_palette_color(3, 0, 0, 255, 255)) {
        syslog(LOG_ERR, "Failed to setup palette colors");
        return 1;
    }

    // Get max resolution for width and height
    camera_width = axoverlay_get_max_resolution_width(1, &error);
    g_error_free(error);
    camera_height = axoverlay_get_max_resolution_height(1, &error);
    g_error_free(error);
    syslog(LOG_INFO, "Max resolution (width x height): %i x %i", camera_width,
            camera_height);

    // Create a large overlay using Palette color space
    struct axoverlay_overlay_data data;
    setup_axoverlay_data(&data);
}

/**
 * brief Main function that starts a stream with different options.
 */
int main(int argc, char** argv) 
{
    SetupOverlay();

    // Hardcode to use three image "color" channels (eg. RGB).
    const unsigned int CHANNELS = 3;

    // Name patterns for the temp file we will create.
    char CONV_PP_FILE_PATTERN[] = "/tmp/larod.pp.test-XXXXXX";
    char CONV_INP_FILE_PATTERN[] = "/tmp/larod.in.test-XXXXXX";
    char CONV_OUT1_FILE_PATTERN[] = "/tmp/larod.out1.test-XXXXXX";
    char CONV_OUT2_FILE_PATTERN[] = "/tmp/larod.out2.test-XXXXXX";

    bool ret = false;
    ImgProvider_t* provider = NULL;
    larodError* error = NULL;
    larodConnection* conn = NULL;
    larodMap* ppMap = NULL;
    larodMap* cropMap = NULL;
    larodModel* ppModel = NULL;
    larodModel* model = NULL;
    larodTensor** ppInputTensors = NULL;
    size_t ppNumInputs = 0;
    larodTensor** ppOutputTensors = NULL;
    size_t ppNumOutputs = 0;
    larodTensor** inputTensors = NULL;
    size_t numInputs = 0;
    larodTensor** outputTensors = NULL;
    size_t numOutputs = 0;
    larodJobRequest* ppReq = NULL;
    larodJobRequest* infReq = NULL;
    void* ppInputAddr = MAP_FAILED;
    void* larodInputAddr = MAP_FAILED;
    void* larodOutput1Addr = MAP_FAILED;
    void* larodOutput2Addr = MAP_FAILED;
    size_t outputBufferSize = 0;
    int larodModelFd = -1;
    int ppInputFd = -1;
    int larodInputFd = -1;
    int larodOutput1Fd = -1;
    int larodOutput2Fd = -1;
    const char* chipString = argv[1];
    const char* modelFile = argv[2];
    const int inputWidth = atoi(argv[3]);
    const int inputHeight = atoi(argv[4]);
    const int numRounds = atoi(argv[5]);
    
    // Open the syslog to report messages for "vdo_larod"
    openlog("vdo_larod", LOG_PID|LOG_CONS, LOG_USER);
    syslog(LOG_INFO, "Starting %s", argv[0]);
    // Register an interrupt handler which tries to exit cleanly if invoked once
    // but exits immediately if further invoked.
    signal(SIGINT, sigintHandler);

    if (argc != 6) {
        syslog(LOG_ERR, "Invalid number of arguments. Required arguments are: "
                        "INF_CHIP MODEL_PATH WIDTH HEIGHT NUM_ROUNDS");
        goto end;
    }

    // Create video stream provider
    unsigned int streamWidth = 0;
    unsigned int streamHeight = 0;
    if (!chooseStreamResolution(inputWidth, inputHeight, &streamWidth,
                                &streamHeight)) {
        syslog(LOG_ERR, "%s: Failed choosing stream resolution", __func__);
        goto end;
    }

    syslog(LOG_INFO, "Creating VDO image provider and creating stream %d x %d",
           streamWidth, streamHeight);
    provider = createImgProvider(streamWidth, streamHeight, 2, VDO_FORMAT_YUV);
    if (!provider) {
        syslog(LOG_ERR, "%s: Could not create image provider", __func__);
        goto end;
    }

    // Calculate crop image
    // 1. The crop area shall fill the input image either horizontally or
    //    vertically.
    // 2. The crop area shall have the same aspect ratio as the output image.
    syslog(LOG_INFO, "Calculate crop image");
    float destWHratio = (float) inputWidth / (float) inputHeight;
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
    ppMap = larodCreateMap(&error);
    if (!ppMap) {
        syslog(LOG_ERR, "Could not create preprocessing larodMap %s", error->msg);
        goto end;
    }
    if (!larodMapSetStr(ppMap, "image.input.format", "nv12", &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }
    if (!larodMapSetIntArr2(ppMap, "image.input.size", streamWidth, streamHeight, &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }
    if(chipString != "ambarella-cvflow"){
        if (!larodMapSetStr(ppMap, "image.output.format", "rgb-interleaved", &error)) {
            syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
            goto end;
        }
    } else {
        if (!larodMapSetStr(ppMap, "image.output.format", "rgb-planar", &error)) {
            syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
            goto end;
        }
    }
    if (!larodMapSetIntArr2(ppMap, "image.output.size", inputWidth, inputHeight, &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }
    cropMap = larodCreateMap(&error);
    if (!cropMap) {
        syslog(LOG_ERR, "Could not create preprocessing crop larodMap %s", error->msg);
        goto end;
    }
    if (!larodMapSetIntArr4(cropMap, "image.input.crop", clipX, clipY, clipW, clipH, &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }

    // Create larod models
    syslog(LOG_INFO, "Create larod models");
    larodModelFd = open(modelFile, O_RDONLY);
    if (larodModelFd < 0) {
        syslog(LOG_ERR, "Unable to open model file %s: %s", modelFile,
               strerror(errno));
        goto end;
    }


    syslog(LOG_INFO, "Setting up larod connection with chip %s and model file %s", chipString, modelFile);
    if (!setupLarod(chipString, larodModelFd, &conn, &model)) {
        goto end;
    }

    // Use libyuv as image preprocessing backend
    const char* larodLibyuvPP = "cpu-proc";
    const larodDevice* dev_pp;
    dev_pp = larodGetDevice(conn, larodLibyuvPP, 0, &error);
    ppModel = larodLoadModel(conn, -1, dev_pp, LAROD_ACCESS_PRIVATE, "", ppMap, &error);
    if (!ppModel) {
            syslog(LOG_ERR, "Unable to load preprocessing model with chip %s: %s", larodLibyuvPP, error->msg);
            goto end;
    } else {
           syslog(LOG_INFO, "Loading preprocessing model with chip %s", larodLibyuvPP);
    }

    // Create input/output tensors
    syslog(LOG_INFO, "Create input/output tensors");
    ppInputTensors = larodCreateModelInputs(ppModel, &ppNumInputs, &error);
    if (!ppInputTensors) {
        syslog(LOG_ERR, "Failed retrieving input tensors: %s", error->msg);
        goto end;
    }
    ppOutputTensors = larodCreateModelOutputs(ppModel, &ppNumOutputs, &error);
    if (!ppOutputTensors) {
        syslog(LOG_ERR, "Failed retrieving output tensors: %s", error->msg);
        goto end;
    }
    inputTensors = larodCreateModelInputs(model, &numInputs, &error);
    if (!inputTensors) {
        syslog(LOG_ERR, "Failed retrieving input tensors: %s", error->msg);
        goto end;
    }
    // This app only supports 1 input tensor right now.
    if (numInputs != 1) {
        syslog(LOG_ERR, "Model has %zu inputs, app only supports 1 input tensor.",
               numInputs);
        goto end;
    }
    outputTensors = larodCreateModelOutputs(model, &numOutputs, &error);
    if (!outputTensors) {
        syslog(LOG_ERR, "Failed retrieving output tensors: %s", error->msg);
        goto end;
    }
    // This app only supports 1 output tensor right now.
    if (numOutputs != 2) {
        syslog(LOG_ERR, "Model has %zu outputs, app only supports 2 output tensors.",
               numOutputs);
        goto end;
    }

    // Determine tensor buffer sizes
    syslog(LOG_INFO, "Determine tensor buffer sizes");
    const larodTensorPitches* ppInputPitches = larodGetTensorPitches(ppInputTensors[0], &error);
    if (!ppInputPitches) {
        syslog(LOG_ERR, "Could not get pitches of tensor: %s", error->msg);
        goto end;
    }
    size_t yuyvBufferSize = ppInputPitches->pitches[0];
    const larodTensorPitches* ppOutputPitches = larodGetTensorPitches(ppOutputTensors[0], &error);
    if (!ppOutputPitches) {
        syslog(LOG_ERR, "Could not get pitches of tensor: %s", error->msg);
        goto end;
    }
    size_t rgbBufferSize = ppOutputPitches->pitches[0];
    size_t expectedSize = inputWidth * inputHeight * CHANNELS;
    if (expectedSize != rgbBufferSize) {
        syslog(LOG_ERR, "Expected video output size %d, actual %d", expectedSize, rgbBufferSize);
        goto end;
    }
    const larodTensorPitches* outputPitches = larodGetTensorPitches(outputTensors[0], &error);
    if (!outputPitches) {
        syslog(LOG_ERR, "Could not get pitches of tensor: %s", error->msg);
        goto end;
    }
    outputBufferSize = outputPitches->pitches[0];

    // Allocate memory for input/output buffers
    syslog(LOG_INFO, "Allocate memory for input/output buffers");
    if (!createAndMapTmpFile(CONV_PP_FILE_PATTERN, yuyvBufferSize,
                             &ppInputAddr, &ppInputFd)) {
        goto end;
    }
    if (!createAndMapTmpFile(CONV_INP_FILE_PATTERN,
                             inputWidth * inputHeight * CHANNELS,
                             &larodInputAddr, &larodInputFd)) {
        goto end;
    }
    if (!createAndMapTmpFile(CONV_OUT1_FILE_PATTERN, 4,
                             &larodOutput1Addr, &larodOutput1Fd)) {
        goto end;
    }
    if (!createAndMapTmpFile(CONV_OUT2_FILE_PATTERN, 4,
                             &larodOutput2Addr, &larodOutput2Fd)) {
        goto end;
    }

    // Connect tensors to file descriptors
    syslog(LOG_INFO, "Connect tensors to file descriptors");
    if (!larodSetTensorFd(ppInputTensors[0], ppInputFd, &error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", error->msg);
        goto end;
    }
    if (!larodSetTensorFd(ppOutputTensors[0], larodInputFd, &error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", error->msg);
        goto end;
    }
    if (!larodSetTensorFd(inputTensors[0], larodInputFd, &error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", error->msg);
        goto end;
    }
    if (!larodSetTensorFd(outputTensors[0], larodOutput1Fd, &error)) {
        syslog(LOG_ERR, "Failed setting output tensor fd: %s", error->msg);
        goto end;
    }

    if (!larodSetTensorFd(outputTensors[1], larodOutput2Fd, &error)) {
        syslog(LOG_ERR, "Failed setting output tensor fd: %s", error->msg);
        goto end;
    }

    // Create job requests
    syslog(LOG_INFO, "Create job requests");
    ppReq = larodCreateJobRequest(ppModel, ppInputTensors, ppNumInputs,
        ppOutputTensors, ppNumOutputs, cropMap, &error);
    if (!ppReq) {
        syslog(LOG_ERR, "Failed creating preprocessing job request: %s", error->msg);
        goto end;
    }

    // App supports only one input/output tensor.
    infReq = larodCreateJobRequest(model, inputTensors, 1, outputTensors,
                                         2, NULL, &error);
    if (!infReq) {
        syslog(LOG_ERR, "Failed creating inference request: %s", error->msg);
        goto end;
    }

    syslog(LOG_INFO, "Start fetching video frames from VDO");
    if (!startFrameFetch(provider)) {
        goto end;
    }

    for (unsigned int i = 0; i < numRounds && !stopRunning; i++) {
        struct timeval startTs, endTs;
        unsigned int elapsedMs = 0;

        // Get latest frame from image pipeline.
        VdoBuffer* buf = getLastFrameBlocking(provider);
        if (!buf) {
            goto end;
        }

        // Get data from latest frame.
        uint8_t* nv12Data = (uint8_t*) vdo_buffer_get_data(buf);

        // Covert image data from NV12 format to interleaved uint8_t RGB format
        gettimeofday(&startTs, NULL);
        memcpy(ppInputAddr, nv12Data, yuyvBufferSize);
        if (!larodRunJob(conn, ppReq, &error)) {
            syslog(LOG_ERR, "Unable to run job to preprocess model: %s (%d)",
                   error->msg, error->code);
            goto end;
        }
        gettimeofday(&endTs, NULL);

        elapsedMs = (unsigned int) (((endTs.tv_sec - startTs.tv_sec) * 1000) +
                                    ((endTs.tv_usec - startTs.tv_usec) / 1000));
        syslog(LOG_INFO, "Converted image in %u ms", elapsedMs);

        // Save the RGB image as a PPM file
        const char* filename = "/tmp/output.ppm";
        saveRgbImageAsPpm(larodInputAddr, inputWidth, inputHeight, filename);

        // Since larodOutputAddr points to the beginning of the fd we should
        // rewind the file position before each job.
        if (lseek(larodOutput1Fd, 0, SEEK_SET) == -1) {
            syslog(LOG_ERR, "Unable to rewind output file position: %s",
                   strerror(errno));
            goto end;
        }

        if (lseek(larodOutput2Fd, 0, SEEK_SET) == -1) {
            syslog(LOG_ERR, "Unable to rewind output file position: %s",
                   strerror(errno));
            goto end;
        }

        gettimeofday(&startTs, NULL);
        if (!larodRunJob(conn, infReq, &error)) {
            syslog(LOG_ERR, "Unable to run inference on model %s: %s (%d)",
                   modelFile, error->msg, error->code);
            goto end;
        }
        gettimeofday(&endTs, NULL);

        elapsedMs = (unsigned int) (((endTs.tv_sec - startTs.tv_sec) * 1000) +
                                    ((endTs.tv_usec - startTs.tv_usec) / 1000));
        syslog(LOG_INFO, "Ran inference for %u ms", elapsedMs);

        if (strcmp(chipString, "ambarella-cvflow") != 0){
            uint8_t* person_pred = (uint8_t*) larodOutput1Addr;
            uint8_t* car_pred = (uint8_t*) larodOutput2Addr;

            syslog(LOG_INFO, "Person detected: %.2f%% - Car detected: %.2f%%",
                (float) person_pred[0] / 2.55f, (float) car_pred[0]  / 2.55f);
        } else {
            uint8_t* car_pred = (uint8_t*) larodOutput1Addr;
            uint8_t* person_pred = (uint8_t*) larodOutput2Addr;
            float float_score_car = *((float*) car_pred);
            float float_score_person  = *((float*) person_pred);
            syslog(LOG_INFO, "Person detected: %.2f%% - Car detected: %.2f%%",
                float_score_person*100, float_score_car*100);
        }

        // Release frame reference to provider.
        returnFrame(provider, buf);
    }

    syslog(LOG_INFO, "Stop streaming video from VDO");
    if (!stopFrameFetch(provider)) {
        goto end;
    }

    ret = true;

end:
    if (provider) {
        destroyImgProvider(provider);
    }
    // Only the model handle is released here. We count on larod service to
    // release the privately loaded model when the session is disconnected in
    // larodDisconnect().
    larodDestroyMap(&ppMap);
    larodDestroyMap(&cropMap);
    larodDestroyModel(&ppModel);
    larodDestroyModel(&model);
    if (conn) {
        larodDisconnect(&conn, NULL);
    }
    if (larodModelFd >= 0) {
        close(larodModelFd);
    }
    if (ppInputAddr != MAP_FAILED) {
        munmap(ppInputAddr, inputWidth * inputHeight * CHANNELS);
    }
    if (ppInputFd >= 0) {
        close(ppInputFd);
    }
    if (larodInputAddr != MAP_FAILED) {
        munmap(larodInputAddr, inputWidth * inputHeight * CHANNELS);
    }
    if (larodInputFd >= 0) {
        close(larodInputFd);
    }
    if (larodOutput1Addr != MAP_FAILED) {
        munmap(larodOutput1Addr, 4);
    }

    if (larodOutput2Addr != MAP_FAILED) {
        munmap(larodOutput2Addr, 4);
    }
    if (larodOutput1Fd >= 0) {
        close(larodOutput1Fd);
    }

    if (larodOutput2Fd >= 0) {
        close(larodOutput2Fd);
    }

    larodDestroyJobRequest(&ppReq);
    larodDestroyJobRequest(&infReq);
    larodDestroyTensors(conn, &inputTensors, numInputs, &error);
    larodDestroyTensors(conn, &outputTensors, numOutputs, &error);
    larodClearError(&error);

    syslog(LOG_INFO, "Exit %s", argv[0]);
    return ret ? EXIT_SUCCESS : EXIT_FAILURE;
}
