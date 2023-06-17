#include "YOLOX.hpp"

YOLOX::YOLOX(size_t streamWidth, size_t streamHeight, const char* device = "cpu-tflite"):
Larod(streamWidth, streamHeight, device)
{
  _grid_strides = generate_grids_and_strides();
}

std::vector<GridAndStride> YOLOX::generate_grids_and_strides()
{

}

void YOLOX::generate_yolox_proposals(float bbox_threshold)
{

}

void YOLOX::nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{

}