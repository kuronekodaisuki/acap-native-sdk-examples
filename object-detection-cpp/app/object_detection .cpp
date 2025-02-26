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

 /**
  * - object_detection -
  *
  * This application loads a larod model which takes an image as input and
  * outputs values corresponding to the class, score and location of detected
  * objects in the image.
  *
  * The application expects eight arguments on the command line in the
  * following order: MODEL WIDTH HEIGHT QUALITY RAW_WIDTH RAW_HEIGHT
  * THRESHOLD LABELSFILE.
  *
  * First argument, MODEL, is a string describing path to the model.
  *
  * Second argument, WIDTH, is an integer for the input width.
  *
  * Third argument, HEIGHT, is an integer for the input height.
  *
  * Fourth argument, QUALITY, is an integer for the desired jpeg quality.
  *
  * Fifth argument, RAW_WIDTH is an integer for camera width resolution.
  *
  * Sixth argument, RAW_HEIGHT is an integer for camera height resolution.
  *
  * Seventh argument, THRESHOLD is an integer ranging from 0 to 100 to select good detections.
  *
  * Eighth argument, LABELSFILE, is a string describing path to the label txt.
  *
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
#include <glib.h>
#include <cairo/cairo.h>
#include <axoverlay.h>

#include "argparse.h"
#include "imgprovider.h"
#include "imgutils.h"
#include "larod.h"
#include "vdo-frame.h"
#include "vdo-types.h"

#define PALETTE_VALUE_RANGE 255.0
static gint overlay_id      = -1;
static gint overlay_id_text = -1;
static gint counter = 10;
static gint top_color = 1;
static gint bottom_color = 3;

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
 * @brief Free up resources held by an array of labels.
 *
 * @param labels An array of label string pointers.
 * @param labelFileBuffer Heap buffer containing the actual string data.
 */
void freeLabels(char** labelsArray, char* labelFileBuffer) {
    free(labelsArray);
    free(labelFileBuffer);
}

/**
 * @brief Reads a file of labels into an array.
 *
 * An array filled by this function should be freed using freeLabels.
 *
 * @param labelsPtr Pointer to a string array.
 * @param labelFileBuffer Pointer to the labels file contents.
 * @param labelsPath String containing the path to the labels file to be read.
 * @param numLabelsPtr Pointer to number which will store number of labels read.
 * @return False if any errors occur, otherwise true.
 */
static bool parseLabels(char*** labelsPtr, char** labelFileBuffer, char *labelsPath,
                 size_t* numLabelsPtr) {
    // We cut off every row at 60 characters.
    const size_t LINE_MAX_LEN = 60;
    bool ret = false;
    char* labelsData = NULL;  // Buffer containing the label file contents.
    char** labelArray = NULL; // Pointers to each line in the labels text.

    struct stat fileStats = {0};
    if (stat(labelsPath, &fileStats) < 0) {
        syslog(LOG_ERR, "%s: Unable to get stats for label file %s: %s", __func__,
               labelsPath, strerror(errno));
        return false;
    }

    // Sanity checking on the file size - we use size_t to keep track of file
    // size and to iterate over the contents. off_t is signed and 32-bit or
    // 64-bit depending on architecture. We just check toward 10 MByte as we
    // will not encounter larger label files and both off_t and size_t should be
    // able to represent 10 megabytes on both 32-bit and 64-bit systems.
    if (fileStats.st_size > (10 * 1024 * 1024)) {
        syslog(LOG_ERR, "%s: failed sanity check on labels file size", __func__);
        return false;
    }

    int labelsFd = open(labelsPath, O_RDONLY);
    if (labelsFd < 0) {
        syslog(LOG_ERR, "%s: Could not open labels file %s: %s", __func__, labelsPath,
               strerror(errno));
        return false;
    }

    size_t labelsFileSize = (size_t) fileStats.st_size;
    // Allocate room for a terminating NULL char after the last line.
    labelsData = malloc(labelsFileSize + 1);
    if (labelsData == NULL) {
        syslog(LOG_ERR, "%s: Failed allocating labels text buffer: %s", __func__,
               strerror(errno));
        goto end;
    }

    ssize_t numBytesRead = -1;
    size_t totalBytesRead = 0;
    char* fileReadPtr = labelsData;
    while (totalBytesRead < labelsFileSize) {
        numBytesRead =
            read(labelsFd, fileReadPtr, labelsFileSize - totalBytesRead);

        if (numBytesRead < 1) {
            syslog(LOG_ERR, "%s: Failed reading from labels file: %s", __func__,
                   strerror(errno));
            goto end;
        }
        totalBytesRead += (size_t) numBytesRead;
        fileReadPtr += numBytesRead;
    }

    // Now count number of lines in the file - check all bytes except the last
    // one in the file.
    size_t numLines = 0;
    for (size_t i = 0; i < (labelsFileSize - 1); i++) {
        if (labelsData[i] == '\n') {
            numLines++;
        }
    }

    // We assume that there is always a line at the end of the file, possibly
    // terminated by newline char. Either way add this line as well to the
    // counter.
    numLines++;

    labelArray = malloc(numLines * sizeof(char*));
    if (!labelArray) {
        syslog(LOG_ERR, "%s: Unable to allocate labels array: %s", __func__,
               strerror(errno));
        ret = false;
        goto end;
    }

    size_t labelIdx = 0;
    labelArray[labelIdx] = labelsData;
    labelIdx++;
    for (size_t i = 0; i < labelsFileSize; i++) {
        if (labelsData[i] == '\n') {
            // Register the string start in the list of labels.
            labelArray[labelIdx] = labelsData + i + 1;
            labelIdx++;
            // Replace the newline char with string-ending NULL char.
            labelsData[i] = '\0';
        }
    }

    // If the very last byte in the labels file was a new-line we just
    // replace that with a NULL-char. Refer previous for loop skipping looking
    // for new-line at the end of file.
    if (labelsData[labelsFileSize - 1] == '\n') {
        labelsData[labelsFileSize - 1] = '\0';
    }

    // Make sure we always have a terminating NULL char after the label file
    // contents.
    labelsData[labelsFileSize] = '\0';

    // Now go through the list of strings and cap if strings too long.
    for (size_t i = 0; i < numLines; i++) {
        size_t stringLen = strnlen(labelArray[i], LINE_MAX_LEN);
        if (stringLen >= LINE_MAX_LEN) {
            // Just insert capping NULL terminator to limit the string len.
            *(labelArray[i] + LINE_MAX_LEN + 1) = '\0';
        }
    }

    *labelsPtr = labelArray;
    *numLabelsPtr = numLines;
    *labelFileBuffer = labelsData;

    ret = true;

end:
    if (!ret) {
        freeLabels(labelArray, labelsData);
    }
    close(labelsFd);

    return ret;
}

/// Set by signal handler if an interrupt signal sent to process.
/// Indicates that app should stop asap and exit gracefully.
volatile sig_atomic_t stopRunning = false;

/**
 * @brief Invoked on SIGINT. Makes app exit cleanly asap if invoked once, but
 * forces an immediate exit without clean up if invoked at least twice.
 *
 * @param sig What signal has been sent. Will always be SIGINT.
 */
void sigintHandler(int sig) {
    // Force an exit if SIGINT has already been sent before.
    if (stopRunning) {
        syslog(LOG_INFO, "Interrupted again, exiting immediately without clean up.");

        exit(EXIT_FAILURE);
    }

    syslog(LOG_INFO, "Interrupted, starting graceful termination of app. Another "
           "interrupt signal will cause a forced exit.");

    // Tell the main thread to stop running inferences asap.
    stopRunning = true;
}


/**
 * @brief Creates a temporary fd and truncated to correct size and mapped.
 *
 * This convenience function creates temp files to be used for input and output.
 *
 * @param fileName Pattern for how the temp file will be named in file system.
 * @param fileSize How much space needed to be allocated (truncated) in fd.
 * @param mappedAddr Pointer to the address of the fd mapped for this process.
 * @param Pointer to the generated fd.
 * @return Positive errno style return code (zero means success).
 */
static bool createAndMapTmpFile(char* fileName, size_t fileSize, void** mappedAddr,
                         int* convFd) {
    syslog(LOG_INFO, "%s: Setting up a temp fd with pattern %s and size %zu", __func__,
            fileName, fileSize);

    int fd = mkstemp(fileName);
    if (fd < 0) {
        syslog(LOG_ERR, "%s: Unable to open temp file %s: %s", __func__, fileName,
                 strerror(errno));
        goto error;
    }

    // Allocate enough space in for the fd.
    if (ftruncate(fd, (off_t) fileSize) < 0) {
        syslog(LOG_ERR, "%s: Unable to truncate temp file %s: %s", __func__, fileName,
                 strerror(errno));
        goto error;
    }

    // Remove since we don't actually care about writing to the file system.
    if (unlink(fileName)) {
        syslog(LOG_ERR, "%s: Unable to unlink from temp file %s: %s", __func__,
                 fileName, strerror(errno));
        goto error;
    }

    // Get an address to fd's memory for this process's memory space.
    void* data =
        mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (data == MAP_FAILED) {
        syslog(LOG_ERR, "%s: Unable to mmap temp file %s: %s", __func__, fileName,
                 strerror(errno));
        goto error;
    }

    *mappedAddr = data;
    *convFd = fd;

    return true;

error:
    if (fd >= 0) {
        close(fd);
    }

    return false;
}

/**
 * @brief Sets up and configures a connection to larod, and loads a model.
 *
 * Opens a connection to larod, which is tied to larodConn. After opening a
 * larod connection the chip specified by larodChip is set for the
 * connection. Then the model file specified by larodModelFd is loaded to the
 * chip, and a corresponding larodModel object is tied to model.
 *
 * larodChip Speficier for which larod chip to use.
 * @param larodModelFd Fd for a model file to load.
 * @param larodConn Pointer to a larod connection to be opened.
 * @param model Pointer to a larodModel to be obtained.
 * @return false if error has occurred, otherwise true.
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
                                 "object_detection", NULL, &error);
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
 * @brief Main function that starts a stream with different options.
 */
int main(int argc, char** argv) 
{
    SetupOverlay();

    // Hardcode to use three image "color" channels (eg. RGB).
    const unsigned int CHANNELS = 3;
    // Hardcode to set output bytes of four tensors from MobileNet V2 model.
    const unsigned int FLOATSIZE = 4;
    const unsigned int TENSOR1SIZE = 80 * FLOATSIZE;
    const unsigned int TENSOR2SIZE = 20 * FLOATSIZE;
    const unsigned int TENSOR3SIZE = 20 * FLOATSIZE;
    const unsigned int TENSOR4SIZE = 1 * FLOATSIZE;

    // Name patterns for the temp file we will create.
    char CONV_INP_FILE_PATTERN[] = "/tmp/larod.in.test-XXXXXX";
    char CONV_PP_FILE_PATTERN[] = "/tmp/larod.pp.test-XXXXXX";
    char CROP_FILE_PATTERN[] = "/tmp/crop.test-XXXXXX";
    char CONV_OUT1_FILE_PATTERN[] = "/tmp/larod.out1.test-XXXXXX";
    char CONV_OUT2_FILE_PATTERN[] = "/tmp/larod.out2.test-XXXXXX";
    char CONV_OUT3_FILE_PATTERN[] = "/tmp/larod.out3.test-XXXXXX";
    char CONV_OUT4_FILE_PATTERN[] = "/tmp/larod.out4.test-XXXXXX";

    bool ret = false;
    ImgProvider_t* provider = NULL;
    ImgProvider_t* provider_raw = NULL;
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
    size_t outputBufferSize = 0;
    int ppInputFd = -1;
    void* larodInputAddr = MAP_FAILED;
    void* cropAddr = MAP_FAILED;
    void* larodOutput1Addr = MAP_FAILED;
    void* larodOutput2Addr = MAP_FAILED;
    void* larodOutput3Addr = MAP_FAILED;
    void* larodOutput4Addr = MAP_FAILED;
    int larodModelFd = -1;
    int larodInputFd = -1;
    int cropFd = -1;
    int larodOutput1Fd = -1;
    int larodOutput2Fd = -1;
    int larodOutput3Fd = -1;
    int larodOutput4Fd = -1;
    char** labels = NULL; // This is the array of label strings. The label
                          // entries points into the large labelFileData buffer.
    size_t numLabels = 0; // Number of entries in the labels array.
    char* labelFileData =
        NULL; // Buffer holding the complete collection of label strings.

    // Open the syslog to report messages for "object_detection"
    openlog("object_detection", LOG_PID|LOG_CONS, LOG_USER);

    args_t args;
    if (!parseArgs(argc, argv, &args)) {
        goto end;
    }

    const char* chipString = args.chip;
    const char* modelFile = args.modelFile;
    const char* labelsFile = args.labelsFile;
    const int inputWidth = args.width;
    const int inputHeight = args.height;
    const int rawWidth = args.raw_width;
    const int rawHeight = args.raw_height;
    const int threshold = args.threshold;
    const int quality = args.quality;


    syslog(LOG_INFO, "Starting %s", argv[0]);
    // Register an interrupt handler which tries to exit cleanly if invoked once
    // but exits immediately if further invoked.
    signal(SIGINT, sigintHandler);

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

    syslog(LOG_INFO, "Creating VDO raw image provider and stream %d x %d",
            rawWidth, rawHeight);
    provider_raw = createImgProvider(rawWidth, rawHeight, 2, VDO_FORMAT_YUV);
    if (!provider_raw) {
      syslog(LOG_ERR, "%s: Could not create raw image provider", __func__);
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


    syslog(LOG_INFO, "Setting up larod connection with chip %s, model %s and label file %s", chipString, modelFile, labelsFile);
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

    outputTensors = larodCreateModelOutputs(model, &numOutputs, &error);
    if (!outputTensors) {
        syslog(LOG_ERR, "Failed retrieving output tensors: %s", error->msg);
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
    const larodTensorPitches* outputPitches = larodGetTensorPitches(outputTensors[0], &error);
    if (!outputPitches) {
        syslog(LOG_ERR, "Could not get pitches of tensor: %s", error->msg);
        goto end;
    }
    outputBufferSize = outputPitches->pitches[0];

    // Allocate space for input tensor
    syslog(LOG_INFO, "Allocate memory for input/output buffers");
    if (!createAndMapTmpFile(CONV_INP_FILE_PATTERN,
                             inputWidth * inputHeight * CHANNELS,
                             &larodInputAddr, &larodInputFd)) {
        goto end;
    }
    if (!createAndMapTmpFile(CONV_PP_FILE_PATTERN, yuyvBufferSize,
                             &ppInputAddr, &ppInputFd)) {
        goto end;
    }
    if (!createAndMapTmpFile(CROP_FILE_PATTERN,
                             rawWidth * rawHeight * CHANNELS,
                             &cropAddr, &cropFd)) {
        goto end;
    }
    if (!createAndMapTmpFile(CONV_OUT1_FILE_PATTERN, TENSOR1SIZE,
                             &larodOutput1Addr, &larodOutput1Fd)) {
        goto end;
    }
    if (!createAndMapTmpFile(CONV_OUT2_FILE_PATTERN, TENSOR2SIZE,
                             &larodOutput2Addr, &larodOutput2Fd)) {
        goto end;
    }

    if (!createAndMapTmpFile(CONV_OUT3_FILE_PATTERN, TENSOR3SIZE,
                             &larodOutput3Addr, &larodOutput3Fd)) {
        goto end;
    }

    if (!createAndMapTmpFile(CONV_OUT4_FILE_PATTERN, TENSOR4SIZE,
                             &larodOutput4Addr, &larodOutput4Fd)) {
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

    syslog(LOG_INFO, "Set input tensors");
    if (!larodSetTensorFd(inputTensors[0], larodInputFd, &error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", error->msg);
        goto end;
    }

    syslog(LOG_INFO, "Set output tensors");
    if (!larodSetTensorFd(outputTensors[0], larodOutput1Fd, &error)) {
        syslog(LOG_ERR, "Failed setting output tensor fd: %s", error->msg);
        goto end;
    }

    if (!larodSetTensorFd(outputTensors[1], larodOutput2Fd, &error)) {
        syslog(LOG_ERR, "Failed setting output tensor fd: %s", error->msg);
        goto end;
    }

    if (!larodSetTensorFd(outputTensors[2], larodOutput3Fd, &error)) {
        syslog(LOG_ERR, "Failed setting output tensor fd: %s", error->msg);
        goto end;
    }

    if (!larodSetTensorFd(outputTensors[3], larodOutput4Fd, &error)) {
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
    infReq = larodCreateJobRequest(model, inputTensors, numInputs, outputTensors,
                                         numOutputs, NULL, &error);
    if (!infReq) {
        syslog(LOG_ERR, "Failed creating inference request: %s", error->msg);
        goto end;
    }

    if (labelsFile) {
        if (!parseLabels(&labels, &labelFileData, labelsFile,
                         &numLabels)) {
            syslog(LOG_ERR, "Failed creating parsing labels file");
            goto end;
        }
    }

    syslog(LOG_INFO, "Found %x input tensors and %x output tensors", numInputs, numOutputs);
    syslog(LOG_INFO, "Start fetching video frames from VDO");
    if (!startFrameFetch(provider)) {
        syslog(LOG_ERR, "Stuck in provider");
        goto end;
    }

    if (!startFrameFetch(provider_raw)) {
        syslog(LOG_ERR, "Stuck in provider raw");
        goto end;
    }

    while (true) 
    {
        struct timeval startTs, endTs;
        unsigned int elapsedMs = 0;

        // Get latest frame from image pipeline.
        VdoBuffer* buf = getLastFrameBlocking(provider);
        if (!buf) {
            syslog(LOG_ERR, "buf empty in provider");
            goto end;
        }

        VdoBuffer* buf_hq = getLastFrameBlocking(provider_raw);
        if (!buf_hq) {
            syslog(LOG_ERR, "buf empty in provider raw");
            goto end;
        }

        // Get data from latest frame.
        uint8_t* nv12Data = (uint8_t*) vdo_buffer_get_data(buf);
        uint8_t* nv12Data_hq = (uint8_t*) vdo_buffer_get_data(buf_hq);

        // Covert image data from NV12 format to interleaved uint8_t RGB format.
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

        if (lseek(larodOutput3Fd, 0, SEEK_SET) == -1) {
            syslog(LOG_ERR, "Unable to rewind output file position: %s",
                   strerror(errno));
            goto end;
        }

        if (lseek(larodOutput4Fd, 0, SEEK_SET) == -1) {
            syslog(LOG_ERR, "Unable to rewind output file position: %s",
                   strerror(errno));
            goto end;
        }

        gettimeofday(&startTs, NULL);
        if (!larodRunJob(conn, infReq, &error)) {
            syslog(LOG_ERR, "Unable to run inference on model %s: %s (%d)",
                   labelsFile, error->msg, error->code);
            goto end;
        }
        gettimeofday(&endTs, NULL);

        elapsedMs = (unsigned int) (((endTs.tv_sec - startTs.tv_sec) * 1000) +
                                    ((endTs.tv_usec - startTs.tv_usec) / 1000));
        syslog(LOG_INFO, "Ran inference for %u ms", elapsedMs);

        float* locations = (float*) larodOutput1Addr;
        float* classes = (float*) larodOutput2Addr;
        float* scores = (float*) larodOutput3Addr;
        float* numberofdetections = (float*) larodOutput4Addr;

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

                unsigned int crop_x = left * rawWidth;
                unsigned int crop_y = top * rawHeight;
                unsigned int crop_w = (right - left) * rawWidth;
                unsigned int crop_h = (bottom - top) * rawHeight;

                if (scores[i] >= args.threshold/100.0)
                {
                    syslog(LOG_INFO, "Object %d: Classes: %s - Scores: %f - Locations: [%f,%f,%f,%f]",
                       i, labels[(int) classes[i]], scores[i], top, left, bottom, right);

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
                }
            }

        }

        // Release frame reference to provider.
        returnFrame(provider, buf);
        returnFrame(provider_raw, buf_hq);
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
    if (provider_raw) {
        destroyImgProvider(provider_raw);
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
    if (larodInputAddr != MAP_FAILED) {
        munmap(larodInputAddr, inputWidth * inputHeight * CHANNELS);
    }
    if (larodInputFd >= 0) {
        close(larodInputFd);
    }
    if (ppInputAddr != MAP_FAILED) {
        munmap(ppInputAddr, inputWidth * inputHeight * CHANNELS);
    }
    if (ppInputFd >= 0) {
        close(ppInputFd);
    }
    if (cropAddr != MAP_FAILED) {
        munmap(cropAddr, rawWidth * rawHeight * CHANNELS);
    }
    if (cropFd >= 0) {
        close(cropFd);
    }
    if (larodOutput1Addr != MAP_FAILED) {
        munmap(larodOutput1Addr, TENSOR1SIZE);
    }

    if (larodOutput2Addr != MAP_FAILED) {
        munmap(larodOutput2Addr, TENSOR2SIZE);
    }

    if (larodOutput3Addr != MAP_FAILED) {
        munmap(larodOutput3Addr, TENSOR3SIZE);
    }

    if (larodOutput4Addr != MAP_FAILED) {
        munmap(larodOutput4Addr, TENSOR4SIZE);
    }

    if (larodOutput1Fd >= 0) {
        close(larodOutput1Fd);
    }

    if (larodOutput2Fd >= 0) {
        close(larodOutput2Fd);
    }

    if (larodOutput3Fd >= 0) {
        close(larodOutput3Fd);
    }

    if (larodOutput4Fd >= 0) {
        close(larodOutput4Fd);
    }
    larodDestroyJobRequest(&ppReq);
    larodDestroyJobRequest(&infReq);
    larodDestroyTensors(conn, &inputTensors, numInputs, &error);
    larodDestroyTensors(conn, &outputTensors, numOutputs, &error);
    larodClearError(&error);

    if (labels) {
        freeLabels(labels, labelFileData);
    }

    syslog(LOG_INFO, "Exit %s", argv[0]);
    return ret ? EXIT_SUCCESS : EXIT_FAILURE;
}
