//
//
//
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "Labels.h"


const char*Labels::operator[](unsigned int index)
{
    if (index < _labels.size())
        return _labels[index].c_str();
    else
        return nullptr;
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
size_t Labels::Load(const char* filename)
{
    // We cut off every row at 60 characters.
    const size_t LINE_MAX_LEN = 60;
    bool ret = false;
    char* labelsData = NULL;  // Buffer containing the label file contents.
    //char** labelArray = NULL; // Pointers to each line in the labels text.

    struct stat fileStats = { 0 };
    if (stat(filename, &fileStats) < 0) {
        syslog(LOG_ERR, "%s: Unable to get stats for label file %s: %s", __func__,
            filename, strerror(errno));
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

    size_t labelsFileSize = (size_t)fileStats.st_size;
    // Allocate room for a terminating NULL char after the last line.
    labelsData = new char[labelsFileSize + 1];
    if (labelsData == NULL) {
        syslog(LOG_ERR, "%s: Failed allocating labels text buffer: %s", __func__,
            strerror(errno));
        return 0;
    }

    int labelsFd = open(filename, O_RDONLY);
    if (labelsFd < 0) {
        syslog(LOG_ERR, "%s: Could not open labels file %s: %s", __func__, filename,
            strerror(errno));
        return 0;
    }

    ssize_t readed = read(labelsFd, labelsData, labelsFileSize);
    close(labelsFd);

    //size_t labelIdx = 0;
    //labelArray[labelIdx] = labelsData;
    //labelIdx++;
    size_t delimiter = 0;
    for (size_t i = 0; i < labelsFileSize; i++)
    {
        if (labelsData[i] == '\n')
        {
            std::string label(&labelsData[delimiter], i - delimiter);
            _labels.push_back(label);
            delimiter = i + 1;
            /*
            // Register the string start in the list of labels.
            labelArray[labelIdx] = labelsData + i + 1;
            labelIdx++;
            // Replace the newline char with string-ending NULL char.
            labelsData[i] = '\0';
            */
        }
    }

    delete[] labelsData;

    syslog(LOG_INFO, "Read %d labels from %s", _labels.size(), filename);
    return _labels.size();
}