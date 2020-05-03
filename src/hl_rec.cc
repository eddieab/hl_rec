#include <filesystem>
#include <fstream>
#include <iostream>

#include "image_io.hh"

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  const fs::path input_directory{argc > 1 ? argv[1] : fs::current_path()};
  std::ofstream outf("timings.txt");

  int total_dng = 0;
  double total_time = 0.;

  for (auto entry = fs::recursive_directory_iterator(input_directory);
       entry != fs::recursive_directory_iterator();
       ++entry) {
    if (entry->is_regular_file() && entry->path().extension() == ".dng") {
      auto f = entry->path();
      auto p = entry->path().parent_path();
      std::string filepath = f.string();
      std::string parent = p.string();
      std::string dng = p.stem().string();

      filepath.erase(remove(filepath.begin(), filepath.end(), '\"'), filepath.end());
      parent.erase(remove(parent.begin(), parent.end(), '\"'), parent.end());
      dng.erase(remove(dng.begin(), dng.end(), '\"'), dng.end());

      std::cout << dng << std::endl;

      auto time = process(filepath, parent);
      total_time += time;
      ++total_dng;

      outf << "dng: " << dng << " | time: " << time << " | fps: " << 1 / time << std::endl;
    }
  }

  std::cout << "Total DNGs: " << total_dng << std::endl;
  std::cout << "Total FPS: " << total_dng / total_time << std::endl;
  outf << "Total DNGs: " << total_dng
       << " | Total Time: " << total_time
       << " | Total FPS: " << total_dng / total_time
       << std::endl;
  outf.close();
}