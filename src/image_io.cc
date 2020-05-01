//
// Created by Eddie Barton on 4/18/20.
//
#include "image_io.hh"

#include "halide_image_io.h"
#include "halide_benchmark.h"
#include "HalideBuffer.h"
#include "camera_pipe.h"

#include "color.h"
#include "tiny_dng_loader.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

static inline unsigned short swap2(unsigned short val) {
  unsigned short ret;

  auto *buf = reinterpret_cast<unsigned char *>(&ret);

  unsigned short x = val;
  buf[1] = static_cast<unsigned char>(x);
  buf[0] = static_cast<unsigned char>(x >> 8);

  return ret;
}

//
// Decode 12bit integer image into 16bit image
//
void decode12(Halide::Runtime::Buffer<float> &im, const unsigned char *data, int width, int
height) {
  int offsets[2][2] = {{0, 1}, {1, 2}};

  int bit_shifts[2] = {4, 0};

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      unsigned char buf[3];

      // Calculate load address for 12bit pixel(three 8 bit pixels)
      int n = int(y * width + x);

      // 24 = 12bit * 2 pixel, 8bit * 3 pixel
      int n2 = n % 2;           // used for offset & bit shifts
      int addr3 = (n / 2) * 3;  // 8bit pixel pos
      int odd = (addr3 % 2);

      int bit_shift = bit_shifts[n2];

      int offset[2];
      offset[0] = offsets[n2][0];
      offset[1] = offsets[n2][1];

      buf[0] = data[addr3 + 0];
      buf[1] = data[addr3 + 1];
      buf[2] = data[addr3 + 2];

      auto b0 = static_cast<unsigned int>(buf[offset[0]] & 0xff);
      auto b1 = static_cast<unsigned int>(buf[offset[1]] & 0xff);

      unsigned int val = (b0 << 8) | b1;
      val = 0xfff & (val >> bit_shift);

      im(x, y) = (float)val;
    }
  }
}

//
// Decode 14bit integer image into 16bit integer image
//
void decode14(Halide::Runtime::Buffer<float> &im, const unsigned char *data, int width, int
height) {
  int offsets[4][3] = {{0, 0, 1}, {1, 2, 3}, {3, 4, 5}, {5, 5, 6}};

  int bit_shifts[4] = {2, 4, 6, 0};

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      unsigned char buf[7];

      // Calculate load addres for 14bit pixel(three 8 bit pixels)
      int n = int(y * width + x);

      // 56 = 14bit * 4 pixel, 8bit * 7 pixel
      int n4 = n % 4;           // used for offset & bitshifts
      int addr7 = (n / 4) * 7;  // 8bit pixel pos
      int odd = (addr7 % 2);

      int offset[3];
      offset[0] = offsets[n4][0];
      offset[1] = offsets[n4][1];
      offset[2] = offsets[n4][2];

      int bit_shift = bit_shifts[n4];
      memcpy(buf, &data[addr7], 7);
      auto b0 = static_cast<unsigned int>(buf[offset[0]] & 0xff);
      auto b1 = static_cast<unsigned int>(buf[offset[1]] & 0xff);
      auto b2 = static_cast<unsigned int>(buf[offset[2]] & 0xff);

      // unsigned int val = (b0 << 16) | (b1 << 8) | b2;
      // unsigned int val = (b2 << 16) | (b0 << 8) | b0;
      unsigned int val = (b0 << 16) | (b1 << 8) | b2;
      // unsigned int val = b2;
      val = 0x3fff & (val >> bit_shift);

      im(x, y) = (float)val;
    }
  }
}

//
// Decode 16bit integer image
//
void decode16(Halide::Runtime::Buffer<float> &im, const unsigned char *data, int width, int
height) {
  const uint16_t *ptr = const_cast<uint16_t *>(reinterpret_cast<const uint16_t *>(data));

#if _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      im(x, y) = (float)ptr[y * width + x];
    }
  }
}

double process(std::string filepath, std::string outdir) {
  Halide::Runtime::Buffer<float> input;

  // load and decode DNG
  std::string warn, err;
  std::vector<tinydng::DNGImage> images;

  // List of custom field infos. This is optional and can be empty.
  std::vector<tinydng::FieldInfo> custom_field_lists;

  // Loads all images(IFD) in the DNG file to `images` array.
  // You can use `LoadDNGFromMemory` API to load DNG image from a memory.
  bool ret = tinydng::LoadDNG(filepath.c_str(), custom_field_lists, &images, &warn, &err);

  if (!warn.empty()) {
    std::cout << "Warn: " << warn << std::endl;
  }

  if (!err.empty()) {
    std::cerr << "Err: " << err << std::endl;
    return EXIT_FAILURE;
  }
  assert(images.size() > 0);
  // Find largest image based on width.
  size_t largest = 0;
  int largest_width = images[0].width;
  for (size_t i = 1; i < images.size(); i++) {
    if (largest_width < images[i].width) {
      largest = i;
      largest_width = images[i].width;
    }
  }

  auto image_idx = static_cast<size_t>(largest);
  const tinydng::DNGImage &image = images[image_idx];
  const int width = image.width;
  int outw = width / 4;
  outw = (8 - outw % 8) + outw;
  const int height = image.height;
  int outh = height / 4;
  outh = (8 - outh % 8) + outh;
  const int bit_depth = image.bits_per_sample;

  auto spp = image.samples_per_pixel;

  const halide_type_t im_type(halide_type_float, 32);
  std::vector<int> im_dimensions = {width, height};

  input = Halide::Runtime::Buffer<float>(im_type, im_dimensions);

  if (bit_depth == 12) {
    decode12(input, &(image.data.at(0)), width, height * spp);
  } else if (bit_depth == 14) {
    decode14(input, &(image.data.at(0)), width, height * spp);
  } else if (bit_depth == 16) {
    decode16(input, &(image.data.at(0)), width, height * spp);
  } else {
    std::cerr << "Unsupported bits_per_sample : " << image.samples_per_pixel << std::endl;
    return EXIT_FAILURE;
  }

  Halide::Runtime::Buffer<uint8_t> output(outw, outh, 3);

  // prep color matrices
  const float A = 1.4388f / 1.435f * 2848.f;
  const float D65 = 1.4388f / 1.438f * 6500.f;
  float cct{}, tint{};
  Halide::Runtime::Buffer<float> wb(3);
  float xyz2cam[3][3], rgb2cam[3][3], cam2rgb[3][3];

  float neutral_xyY[3] = {1.f, 1.f, 1.f};

  color::neutral_to_xyY(image.as_shot_neutral,
                        neutral_xyY,
                        image.calibration_illuminant1,
                        image.color_matrix1,
                        image.color_matrix2);

  color::xyY_to_CCT_tint(neutral_xyY, cct, tint);

  for (int i{}; i < 3; ++i)
    wb(i) = image.as_shot_neutral[1] / image.as_shot_neutral[i];

  if (image.calibration_illuminant1 == tinydng::LIGHTSOURCE_STANDARD_LIGHT_A) {
    color::CCT_to_CM(cct, A, D65, image.color_matrix1, image.color_matrix2, xyz2cam);
  } else {
    color::CCT_to_CM(cct, A, D65, image.color_matrix2, image.color_matrix1, xyz2cam);
  }
  color::mat_33_33_mult(xyz2cam, color::srgb2xyz, rgb2cam);

  float scale[3]
      {1.f / (rgb2cam[0][0] + rgb2cam[0][1] + rgb2cam[0][2]),
       1.f / (rgb2cam[1][0] + rgb2cam[1][1] + rgb2cam[1][2]),
       1.f / (rgb2cam[2][0] + rgb2cam[2][1] + rgb2cam[2][2])};

  for (int j = 0; j < 3; ++j) {
    rgb2cam[0][j] *= scale[0];
    rgb2cam[1][j] *= scale[1];
    rgb2cam[2][j] *= scale[2];
  }

  color::mat_inv(rgb2cam, cam2rgb);

  Halide::Runtime::Buffer<float> cm(3, 3), icm(3, 3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      cm(i, j) = cam2rgb[i][j];
      icm(i, j) = rgb2cam[i][j];
    }

  float blackLevel = image.black_level[0];
  float whiteLevel = image.white_level[0];

  // get cfa pattern
  Halide::Runtime::Buffer<int> cfa(2, 2);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      if (image.cfa_pattern[i][j] == 0)
        cfa(0, 0) = 2 * i + j;
      if (image.cfa_pattern[i][j] == 2)
        cfa(1, 1) = 2 * i + j;
    }
  }
  cfa(0, 1) = cfa(0, 0) % 2 == 1 ? cfa(0, 0) - 1 : cfa(0, 0) + 1;
  cfa(1, 0) = cfa(1, 1) % 2 == 1 ? cfa(1, 1) - 1 : cfa(1, 1) + 1;

  // begin benchmarks returning average
  std::string outpath = outdir;
  std::string file;
  double best;
  double avg = 0.;
  bool hsv = true;
  bool hdr = false;
  bool log = true;

  for (float gain = 1.f; gain < 8.f; gain *= 2.f) {
    best = Halide::Tools::benchmark(1, 1, [&]() {
      camera_pipe(input,
                  wb, cm, icm, cfa, blackLevel, whiteLevel, gain,
                  hsv, hdr, log,
                  output);
      output.device_sync();
    });
    avg += best;
    file = outdir + "/sdr_log_+" + std::to_string(int(log2(gain))) + "ev.png";
    Halide::Tools::save_image(output, file);

    hdr = true;
    best = Halide::Tools::benchmark(1, 1, [&]() {
      camera_pipe(input,
                  wb, cm, icm, cfa, blackLevel, whiteLevel, gain,
                  hsv, hdr, log,
                  output);
      output.device_sync();
    });
    avg += best;
    file = outdir + "/hsv_+" + std::to_string(int(log2(gain))) + "ev.png";
    Halide::Tools::save_image(output, file);

    hsv = false;
    best = Halide::Tools::benchmark(1, 1, [&]() {
      camera_pipe(input,
                  wb, cm, icm, cfa, blackLevel, whiteLevel, gain,
                  hsv, hdr, log,
                  output);
      output.device_sync();
    });
    avg += best;
    file = outdir + "/lch_+" + std::to_string(int(log2(gain))) + "ev.png";
    Halide::Tools::save_image(output, file);
  }

  return avg / 9.;
}
