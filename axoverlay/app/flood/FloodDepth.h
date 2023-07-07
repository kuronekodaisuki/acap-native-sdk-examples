#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

float Depth(uint8_t* yuv, int width, int height);
bool Initialize(const char* filename, int width, int height);
void Release();

#ifdef __cplusplus
}
#endif