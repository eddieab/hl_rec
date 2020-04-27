#include "halide_benchmark.h"

#include "camera_pipe.h"
#ifndef NO_AUTO_SCHEDULE
#include "camera_pipe_auto_schedule.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "color.h"
#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "image_io.hh"
#include "tiny_dng_loader.h"

int main(int argc, char **argv) {
  std::string input_filename = "colorchart.dng";

  if (argc > 1) {
    input_filename = std::string(argv[1]);
  }

  Halide::Runtime::Buffer<float> input;

  // load and decode DNG
  std::string warn, err;
  std::vector<tinydng::DNGImage> images;

  // List of custom field infos. This is optional and can be empty.
  std::vector<tinydng::FieldInfo> custom_field_lists;

  // Loads all images(IFD) in the DNG file to `images` array.
  // You can use `LoadDNGFromMemory` API to load DNG image from a memory.
  bool ret = tinydng::LoadDNG(input_filename.c_str(), custom_field_lists, &images, &warn, &err);

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

  Halide::Runtime::Buffer<float> cm(3, 3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      cm(i, j) = cam2rgb[i][j];

  int timing_iterations = 1000;
  float blackLevel = image.black_level[0];
  float whiteLevel = image.white_level[0];

  Halide::Runtime::Buffer<int> cfa(2, 2);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      if (image.cfa_pattern[i][j] == 0)
        cfa(0, 0) = 2 * i + j;
      if (image.cfa_pattern[i][j] == 2)
        cfa(1, 1) = 2 * i + j;
    }

  cfa(0, 1) = cfa(0, 0) % 2 == 1 ? cfa(0, 0) - 1 : cfa(0, 0) + 1;
  cfa(1, 0) = cfa(1, 1) % 2 == 1 ? cfa(1, 1) - 1 : cfa(1, 1) + 1;

  double best;
  bool hdr = false;
  bool log = true;

  best = Halide::Tools::benchmark(timing_iterations, 1, [&]() {
    camera_pipe(input, wb, cm, cfa, blackLevel, whiteLevel, hdr, log, output);
    output.device_sync();
  });
  fprintf(stderr, "Halide (manual):\t%gus\n", best * 1e6);

#ifndef NO_AUTO_SCHEDULE
  best = Halide::Tools::benchmark(timing_iterations, 1, [&]() {
    camera_pipe(input, wb, cm, cfa, blackLevel, whiteLevel, hdr, log, output);
    output.device_sync();
  });
  fprintf(stderr, "Halide (auto):\t%gus\n", best * 1e6);
#endif

  fprintf(stderr, "output: %s\n", "sdr.png");
  Halide::Tools::save_image(output, "sdr.png");
  fprintf(stderr, "        %d %d\n", output.width(), output.height
  ());

  return EXIT_SUCCESS;
}