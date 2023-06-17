#include "Larod.hpp"

#define NMS_THRESHOLD 0.45;
#define BBOX_CONF_THRESHOLD 0.3

class Object
{
public:
    int label;
    float prob;

    bool operator<(const Object& right) const
    {
        return prob > right.prob;
    }
};

class YOLOX: public Larod
{
public:
    YOLOX(size_t streamWidth, size_t streamHeight, const char* device = "cpu-tflite");
    ~YOLOX();

private:
    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;

        GridAndStride(int g0, int g1, int s): grid0(g0), grid1(g1), stride(s) {}
    };

    float _nms_threshold = NMS_THRESHOLD;
    float _bbox_confidential_threshold = BBOX_CONF_THRESHOLD;
    size_t _numClasses;
    std::vector<Object> _proposals;
    std::vector<Object> _objects;
    std::vector<GridAndStride> _grid_strides;

    std::vector<GridAndStride> generate_grids_and_strides();
    void generate_yolox_proposals(float bbox_threshold);
    void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);


}