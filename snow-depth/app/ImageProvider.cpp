#include "ImageProvider.hpp"

#include <assert.h>
#include <errno.h>
#include <gmodule.h>
#include <syslog.h>
#include <vdo-channel.h>

#include "vdo-map.h"

#define VDO_CHANNEL (1)

ImageProvider::ImageProvider(unsigned int w, unsigned int h,
                unsigned int numFrames, VdoFormat vdoFormat)
{
    bool mtxInitialized = false;
    bool condInitialized = false;

    vdoFormat = format;
    numAppFrames = numFrames;

    if (pthread_mutex_init(&frameMutex, NULL)) {
        syslog(LOG_ERR, "%s: Unable to initialize mutex: %s", __func__,
                 strerror(errno));
        goto errorExit;
    }
    mtxInitialized = true;

    if (pthread_cond_init(&frameDeliverCond, NULL)) {
        syslog(LOG_ERR, "%s: Unable to initialize condition variable: %s", __func__,
                 strerror(errno));
        goto errorExit;
    }
    condInitialized = true;

    deliveredFrames = g_queue_new();
    if (!deliveredFrames) {
        syslog(LOG_ERR, "%s: Unable to create deliveredFrames queue!", __func__);
        goto errorExit;
    }

    processedFrames = g_queue_new();
    if (!processedFrames) {
        syslog(LOG_ERR, "%s: Unable to create processedFrames queue!", __func__);
        goto errorExit;
    }

    if (!createStream(w, h)) {
        syslog(LOG_ERR, "%s: Could not create VDO stream!", __func__);
        goto errorExit;
    }
}

ImageProvider::~ImageProvider()
{
    releaseVdoBuffers();

    pthread_mutex_destroy(&frameMutex);
    pthread_cond_destroy(&frameDeliverCond);

    g_queue_free(deliveredFrames);
    g_queue_free(processedFrames);
}

bool ImageProvider::createStream(unsigned int w, unsigned int h)
{
    VdoMap* vdoMap = vdo_map_new();
    GError* error = NULL;
    bool ret = false;

    if (!vdoMap) {
        syslog(LOG_ERR, "%s: Failed to create vdo_map", __func__);
        goto end;
    }

    vdo_map_set_uint32(vdoMap, "channel", VDO_CHANNEL);
    vdo_map_set_uint32(vdoMap, "format", vdoFormat);
    vdo_map_set_uint32(vdoMap, "width", w);
    vdo_map_set_uint32(vdoMap, "height", h);
    // We will use buffer_alloc() and buffer_unref() calls.
    vdo_map_set_uint32(vdoMap, "buffer.strategy", VDO_BUFFER_STRATEGY_EXPLICIT);

    syslog(LOG_INFO, "Dump of vdo stream settings map =====");
    vdo_map_dump(vdoMap);

    VdoStream* vdoStream = vdo_stream_new(vdoMap, NULL, &error);
    if (!vdoStream) {
        syslog(LOG_ERR, "%s: Failed creating vdo stream: %s", __func__,
                 (error != NULL) ? error->message : "N/A");
        goto errorExit;
    }

    if (!allocateVdoBuffers()) {
        syslog(LOG_ERR, "%s: Failed setting up VDO buffers!", __func__);
        goto errorExit;
    }

    // Start the actual VDO streaming.
    if (!vdo_stream_start(vdoStream, &error)) {
        syslog(LOG_ERR, "%s: Failed starting stream: %s", __func__,
                 (error != NULL) ? error->message : "N/A");
        goto errorExit;
    }

    //provider->vdoStream = vdoStream;
}

bool ImageProvider::allocateVdoBuffers()
{
    GError* error = NULL;
    bool ret = false;

    for (size_t i = 0; i < NUM_VDO_BUFFERS; i++)
    {
        vdoBuffers[i] = vdo_stream_buffer_alloc(vdoStream, NULL, &error);
        if (vdoBuffers[i] == NULL) {
            syslog(LOG_ERR, "%s: Failed creating VDO buffer: %s", __func__,
                     (error != NULL) ? error->message : "N/A");
            goto errorExit;
        }

        // Make a 'speculative' vdo_buffer_get_data() call to trigger a
        // memory mapping of the buffer. The mapping is cached in the VDO
        // implementation.
        void* dummyPtr = vdo_buffer_get_data(vdoBuffers[i]);
        if (!dummyPtr) {
            syslog(LOG_ERR, "%s: Failed initializing buffer memmap: %s", __func__,
                     (error != NULL) ? error->message : "N/A");
            goto errorExit;
        }

        if (!vdo_stream_buffer_enqueue(vdoStream, vdoBuffers[i],
                                       &error)) {
            syslog(LOG_ERR, "%s: Failed enqueue VDO buffer: %s", __func__,
                     (error != NULL) ? error->message : "N/A");
            goto errorExit;
        }
    }
    return true;
}

void ImageProvider::releaseVdoBuffers()
{
    if (vdoStream)
    {
        for (size_t i = 0; i < NUM_VDO_BUFFERS; i++)
        {
            if (vdoBuffers[i] != NULL)
            {
                vdo_stream_buffer_unref(vdoStream, &vdoBuffers[i], NULL);
            }
        }
    }
}

bool ImageProvider::chooseStreamResolution(unsigned int reqWidth, unsigned int reqHeight,
                            unsigned int* chosenWidth,
                            unsigned int* chosenHeight)
{
    VdoResolutionSet* set = NULL;
    VdoChannel* channel = NULL;
    GError* error = NULL;
    bool ret = false;

    assert(chosenWidth);
    assert(chosenHeight);

    // Retrieve channel resolutions
    channel = vdo_channel_get(VDO_CHANNEL, &error);
    if (!channel) {
        syslog(LOG_ERR, "%s: Failed vdo_channel_get(): %s", __func__,
                 (error != NULL) ? error->message : "N/A");
    g_clear_object(&channel);
    g_free(set);
    g_clear_error(&error);
    }
    // We filter on resolutions that are supported for VDO_FORMAT_YUV
    g_autoptr(VdoMap) filter = vdo_map_new();
    vdo_map_set_uint32(filter, "format", VDO_FORMAT_YUV);
    vdo_map_set_string(filter, "select", "all");
    set = vdo_channel_get_resolutions(channel, filter, &error);
    if (!set) {
        syslog(LOG_ERR, "%s: Failed vdo_channel_get_resolutions(): %s", __func__,
                 (error != NULL) ? error->message : "N/A");
    g_clear_object(&channel);
    g_free(set);
    g_clear_error(&error);
    }

    // Find smallest VDO stream resolution that fits the requested size.
    ssize_t bestResolutionIdx = -1;
    unsigned int bestResolutionArea = UINT_MAX;
    for (ssize_t i = 0; (gsize) i < set->count; ++i) {
        VdoResolution* res = &set->resolutions[i];
        if ((res->width >= reqWidth) && (res->height >= reqHeight)) {
            unsigned int area = res->width * res->height;
            if (area < bestResolutionArea) {
                bestResolutionIdx = i;
                bestResolutionArea = area;
            }
        }
    }

    // If we got a reasonable w/h from the VDO channel info we use that
    // for creating the stream. If that info for some reason was empty we
    // fall back to trying to create a stream with client-supplied w/h.
    *chosenWidth = reqWidth;
    *chosenWidth = reqHeight;
    if (bestResolutionIdx >= 0) {
        *chosenWidth = set->resolutions[bestResolutionIdx].width;
        *chosenHeight = set->resolutions[bestResolutionIdx].height;
        syslog(LOG_INFO, "%s: We select stream w/h=%u x %u based on VDO channel info.\n",
                __func__, *chosenWidth, *chosenHeight);
    } else {
        syslog(LOG_WARNING, "%s: VDO channel info contains no reslution info. Fallback "
                   "to client-requested stream resolution.",
                   __func__);
    }

    ret = true;

    return ret;
}
