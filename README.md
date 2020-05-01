## Foundational Information

This program is a testbed for various highlight reconstruction algorithms from bayer pattern camera raw data. This program uses the [camera_pipe](https://github.com/halide/Halide/tree/master/apps/camera_pipe) app from [Halide](https://github.com/halide/Halide) with modifications as the main processor. [tinydngloader](https://github.com/syoyo/tinydngloader) is used for decoding the image and parsing the metadata necessary for color transforms. miniz is included in the tinydngloader repo and is required for decoding losslessly compressed DNGs. For image output you'll need libjpeg, libpng, and/or libtiff.

## Installation

This project was written using [Halide](https://github.com/halide/Halide/commit/e05e957e41655ab9c5ab7c372fee866b7df0106b) and [tinydngloader](https://github.com/syoyo/tinydngloader/commit/1acccbb4a6dd164a4296663ef90ac212877acaa1). This has been tested only with these linked commits. There have been updates to both repos that have not been integrated here.

Follow the instructions for building Halide from scratch for your system. If on macOS Catalina, the Xcode clang installation suffices. Windows and Linux will require installing [LLVM 9](https://releases.llvm.org/download.html#9.0.1).

After cloning and building the necessary repositories, you can edit the CMakeLists.txt file environment variables. Run cmake to generate the build scripts then use make to build the app.

## Usage

The program takes a DNG file as an argument and outputs an SDR version, LCh reconstructed version, and HSV reconstructed version of the processed image.

    hl_rec /path/to/image.dng

## TODO

- Add full directory processing
- Add benchmarking for multiple images
- Fix the input/output dimension mismatch
- Clean up