//
//
//
#define _USE_MATH_DEFINES
#include "math.h"
#include "RotateAndCrop.h"

/// <summary>
/// �o�񎟕��
/// </summary>
/// <param name="src">���摜</param>
/// <param name="X">���W</param>
/// <param name="Y">���W</param>
/// <returns>��f</returns>
uchar Bilinear8(cv::Mat src, double X, double Y)
{
	int x = (int)floor(X);
	int y = (int)floor(Y);
	double fX = X - x;
	double fY = Y - y;
	uchar* ptr0 = src.ptr(y);
	uchar* ptr1 = src.ptr(y + 1);

	// ��f����
	double pixel
		= (ptr0[x] * (1 - fX) + ptr0[x + 1] * fX) * (1 - fY)
		+ (ptr1[x] * (1 - fX) + ptr1[x + 1] * fX) * fY;

	return (uchar)pixel;
}

/// <summary>
/// �o�񎟕��
/// </summary>
/// <param name="src">���摜</param>
/// <param name="X">���W</param>
/// <param name="Y">���W</param>
/// <returns>��f</returns>
cv::Vec3b Bilinear24(cv::Mat src, double X, double Y)
{
	int x = (int)floor(X);
	int y = (int)floor(Y);
	double fX = X - x;
	double fY = Y - y;
	uchar* ptr0 = src.ptr(y, x);
	uchar* ptr1 = src.ptr(y + 1, x);

	// ��f����
	double b
		= (ptr0[0] * (1 - fX) + ptr0[3] * fX) * (1 - fY)
		+ (ptr1[0] * (1 - fX) + ptr1[3] * fX) * fY;
	double g
		= (ptr0[0] * (1 - fX) + ptr0[3] * fX) * (1 - fY)
		+ (ptr1[0] * (1 - fX) + ptr1[3] * fX) * fY;
	double r
		= (ptr0[0] * (1 - fX) + ptr0[3] * fX) * (1 - fY)
		+ (ptr1[0] * (1 - fX) + ptr1[3] * fX) * fY;
	return cv::Vec3b((uchar)b, (uchar)g, (uchar)r);
}

/// <summary>
/// �摜�̉�]
/// </summary>
/// <param name="src">���摜</param>
/// <param name="dest"></param>
/// <param name="cx">��]���S</param>
/// <param name="cy">��]���S</param>
/// <param name="degree">��]�p�x</param>
/// <param name="size"></param>
/// <returns></returns>
bool RotateAndCrop(cv::Mat src, cv::Mat dest, float cx, float cy, double degree)
{
	double radian = M_PI / 180 * degree;
	double c = cos(radian);
	double s = sin(radian);

	int dx = dest.cols / 2;
	int dy = dest.rows / 2;

	// ���摜�̊J�n���W
	double X = cx + (s * dy - c * dx);
	double Y = cy - (c * dy + s * dx);

	switch (src.type())
	{
	case CV_8UC1:
		// �ϊ���̍��W
		for (int y = 0; y < dest.rows; y++)
		{
			uchar* ptr = dest.ptr(y);
			double Xx = X;
			double Yy = Y;
			for (int x = 0; x < dest.cols; x++)
			{
				ptr[x] = Bilinear8(src, Xx, Yy);

				// ���W�X�V
				Xx += c;
				Yy += s;
			}
			// ���W�X�V
			X -= s;
			Y += c;
		}
		return true;

	case CV_8UC3:
		// �ϊ���̍��W
		for (int y = 0; y < dest.rows; y++)
		{
			uchar* ptr = dest.ptr(y);
			double Xx = X;
			double Yy = Y;
			for (int x = 0; x < dest.cols; x++)
			{
				cv::Vec3b pixel = Bilinear24(src, Xx, Yy);
				ptr[x * 3] = pixel[0];
				ptr[x * 3 + 1] = pixel[1];
				ptr[x * 3 + 2] = pixel[2];

				// ���W�X�V
				Xx += c;
				Yy += s;
			}
			// ���W�X�V
			X -= s;
			Y += c;
		}
		return true;

	}
	return false;
}

/// <summary>
/// ��]
/// </summary>
/// <param name="src"></param>
/// <param name="cx"></param>
/// <param name="cy"></param>
/// <param name="degree"></param>
/// <param name="width"></param>
/// <param name="height"></param>
/// <returns></returns>
cv::Mat RotateAndCrop(cv::Mat src, float cx, float cy, double degree, int width, int height)
{
	cv::Mat dest;
	dest.create(height, width, src.type());

	RotateAndCrop(src, dest, cx, cy, degree);

	return dest;
}