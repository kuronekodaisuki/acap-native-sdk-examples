#include <vector>
#include "Larod.hpp"

#define NMS_THRESHOLD 0.45;
#define BBOX_CONF_THRESHOLD 0.3

class Object
{
public:
    int label;
    float prob;
    float x, y;
    float width, height;

    bool operator<(const Object& right) const
    {
        return prob > right.prob;
    }

    inline float Right() const {return x + width;}
    inline float Bottom() const {return y + height;}

    /// @brief Area of intersection
    /// @param obj
    /// @return
    float IntersectionArea(const Object& obj) const
    {
        float left, top;
        float right = std::min(Right(), obj.Right());
        float bottom = std::min(Bottom(), obj.Bottom());
        if (x <= obj.x && obj.x < Right())
        {
            left = obj.x;
            if (y <= obj.y && obj.y < Bottom())
            {
                top = obj.y;
                return (right - left) * (bottom - top);
            }
            else if (obj.y <= y && y < obj.Bottom())
            {
                top = y;
                return (right - left) * (bottom - top);
            }
        }
        else if (obj.x <= x && x <= obj.Right())
        {
            left = x;
            if (y <= obj.y && obj.y < Bottom())
            {
                top = obj.y;
                return (right - left) * (bottom - top);
            }
            else if (obj.y <= y && y < obj.Bottom())
            {
                top = y;
                return (right - left) * (bottom - top);
            }
        }
        return 0;
    }
};

class YOLOX: public Larod
{
public:
    /// @brief Constructor
    /// @param streamWidth
    /// @param streamHeight
    /// @param device
    YOLOX(size_t streamWidth, size_t streamHeight, const char* device = "cpu-tflite");

    /// @brief Destructor
    ~YOLOX();

    size_t LoadLabels(const char* filename);

    /// @brief Load YOLOX model
    /// @param filename
    /// @param width
    /// @param height
    /// @param channels
    /// @param modelname
    /// @return
    bool LoadModel(const char* filename, size_t width, size_t height, size_t channels = 3, const char* modelname = "inference");

    /// @brief Do inference
    /// @param yuvData
    /// @return
    bool DoInference(u_char* data);

    /// @brief Post process
    /// @param width
    /// @param height
    /// @param scaleX
    /// @param scaleY
    bool PostProcess();

private:
    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;

        GridAndStride(int g0, int g1, int s): grid0(g0), grid1(g1), stride(s) {}
    };
    float* _output;
    float _nms_threshold = NMS_THRESHOLD;
    float _bbox_confidential_threshold = BBOX_CONF_THRESHOLD;
    size_t _numClasses;
    std::vector<Object> _proposals;
    std::vector<Object> _objects;
    std::vector<GridAndStride> _grid_strides;

    std::vector<GridAndStride> generate_grids_and_strides(size_t width, size_t height);
    void generate_yolox_proposals(float bbox_threshold);
    void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold);
};