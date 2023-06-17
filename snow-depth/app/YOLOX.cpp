#include <algorithm>
#include <math.h>
#include "YOLOX.hpp"

YOLOX::YOLOX(size_t streamWidth, size_t streamHeight, const char* device):
Larod(streamWidth, streamHeight, device)
{
  _grid_strides = generate_grids_and_strides();
}

YOLOX::~YOLOX()
{

}

void YOLOX::postProcess(const int width, const int height, float scaleX, float scaleY)
{
    _output = (float*)_outputs[0].GetPtr();
    _proposals.clear();
    generate_yolox_proposals(_bbox_confidential_threshold);

    if (2 <= _proposals.size())
    {
        std::sort(_proposals.begin(), _proposals.end());
    }

    std::vector<int> picked;
    nms_sorted_bboxes(_proposals, picked, _nms_threshold);

    size_t count = picked.size();

    _objects.resize(count);
    for (size_t i = 0; i < count; i++)
    {
        _objects[i] = _proposals[picked[i]];

        _objects[i].x /= scaleX;
        _objects[i].y /= scaleY;
        _objects[i].width /= scaleX;
        _objects[i].height /= scaleY;
    }
}

std::vector<YOLOX::GridAndStride> YOLOX::generate_grids_and_strides()
{
    std::vector<int> strides = { 8, 16, 32 };

    std::vector<GridAndStride> grid_strides;
    for (auto stride : strides)
    {
        int num_grid_y = _modelHeight / stride;
        int num_grid_x = _modelWidth / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                grid_strides.push_back(GridAndStride(g0, g1, stride));
            }
        }
    }
    _numClasses = _labels.GetCount();

    return grid_strides;
}

void YOLOX::generate_yolox_proposals(float prob_threshold)
{
    const size_t num_anchors = _grid_strides.size();

    for (size_t anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = _grid_strides[anchor_idx].grid0;
        const int grid1 = _grid_strides[anchor_idx].grid1;
        const int stride = _grid_strides[anchor_idx].stride;

        const size_t offset = anchor_idx * (_numClasses + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (_output[offset + 0] + grid0) * stride;
        float y_center = (_output[offset + 1] + grid1) * stride;
        float w = exp(_output[offset + 2]) * stride;
        float h = exp(_output[offset + 3]) * stride;
        float box_objectness = _output[offset + 4];

        float x0 = x_center - w / 2;
        float y0 = y_center - h / 2;

        for (size_t class_idx = 0; class_idx < _numClasses; class_idx++)
        {
            float box_cls_score = _output[offset + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (prob_threshold < box_prob)
            {
                Object obj;
                obj.x = x0;
                obj.y = y0;
                obj.width = w;
                obj.height = h;
                obj.label = (int)class_idx;
                obj.prob = box_prob;

                _proposals.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}

void YOLOX::nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const size_t n = objects.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++)
    {
        areas[i] = objects[i].width * objects[i].height;
    }

    for (size_t i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            // intersection over union
            float inter_area = a.IntersectionArea(b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (nms_threshold < inter_area / union_area)
                keep = 0;
        }

        if (keep)
            picked.push_back((int)i);
    }
}