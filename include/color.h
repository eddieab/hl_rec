// D16 color space and color temperature adjustment functions
// use C++11 compiler

#ifndef BOLEXSDK_MODULES_COLOR_H
#define BOLEXSDK_MODULES_COLOR_H

#include <array>

namespace color {
const float eps = 216.f / 24389.f;
const float kappa = 24389.f / 27.f;

const float srgb2xyz[3][3] = {
    {0.4124564f, 0.3575761f, 0.1804375f},
    {0.2126729f, 0.7151522f, 0.0721750f},
    {0.0193339f, 0.1191920f, 0.9503041f}
};

const float xyz2srgb[3][3] = {
    {3.2404542f, -1.5371385f, -0.4985314f},
    {-0.9692660f, 1.8760108f, 0.0415560f},
    {0.0556434f, -0.2040259f, 1.0572252f}
};

const float D50[3]{0.34567f, 0.35850f, 1.00000f};

const float xyz_d65[3] = {0.95047f, 1.f, 1.08883f};

const float illuminant_standard_light_A = 1.4388f / 1.435f * 2848.f;
const float illuminant_D65 = 1.4388f / 1.438f * 6500.f;

void mat_33_33_mult(float[3][3], const float[3][3], float[3][3]);
void mat_33_31_mult(const float[3][3], const double[3], float[3]);
void mat_inv(float[3][3], float[3][3]);

void XYZ_to_xyY(float[3], float[3]);

void xyY_to_CCT_tint(float[3], float &, float &);

void CCT_to_CM(const float, const float, const float, double const[3][3], double const[3][3],
               float[3][3]);

void neutral_to_xyY(double const[3], float[3], int, double const [3][3], double const [3][3]);
}

#endif