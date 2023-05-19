#pragma once

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>

#include "vdo-stream.h"
#include "vdo-types.h"

#define NUM_VDO_BUFFERS (8)

class ImageProvider
{
public:
    ImageProvider(unsigned int w, unsigned int h,
                unsigned int numFrames, VdoFormat vdoFormat);
    ~ImageProvider();

    bool StartFetch();
    void StopFetch();
    VdoBuffer* Get();
    void Put(VdoBuffer* buffer);

    bool chooseStreamResolution(unsigned int reqWidth, unsigned int reqHeight,
                            unsigned int* chosenWidth,
                            unsigned int* chosenHeight);

protected:
    bool createStream(unsigned int w, unsigned int h);
    bool allocateVdoBuffers();
    void releaseVdoBuffers();

private:
    /// Stream configuration parameters.
    VdoFormat vdoFormat;

    /// Vdo stream and buffers handling.
    VdoStream* vdoStream;
    VdoBuffer* vdoBuffers[NUM_VDO_BUFFERS];

    /// Keeping track of frames' statuses.
    GQueue* deliveredFrames;
    GQueue* processedFrames;
    /// Number of frames to keep in the deliveredFrames queue.
    unsigned int numAppFrames;

    /// To support fetching frames asynchonously with VDO.
    pthread_mutex_t frameMutex;
    pthread_cond_t frameDeliverCond;
    pthread_t fetcherThread;
    atomic_bool shutDown;
}