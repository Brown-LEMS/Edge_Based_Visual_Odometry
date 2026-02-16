#!/bin/bash
clear

module_loaded() {
    module list 2>&1 | grep -q "$1"
}

[[ $(module_loaded cmake/3.29.6-ocf3) ]] || module load cmake/3.29.6-ocf3
[[ $(module_loaded eigen/3.4.0-hggg) ]] || module load eigen/3.4.0-hggg
[[ $(module_loaded zlib/1.3.1-xo4x) ]] || module load zlib/1.3.1-xo4x
[[ $(module_loaded zstd/1.5.7-v3mt) ]] || module load zstd/1.5.7-v3mt
[[ $(module_loaded boost/1.88.0-6ijj) ]] || module load boost/1.88.0-6ijj
[[ $(module_loaded opencv) ]] || module load opencv

# Build step
rm -r -f build
mkdir -p build && cd build
cmake ..
make -j

cd ..
echo "Build completed successfully."
echo "Run the application config/eth3d_delivery_area.yaml"
./bin/main_VO --config_file config/eth3d_delivery_area.yaml