#!/bin/bash
clear

module_loaded() {
    module list 2>&1 | grep -q "$1"
}

[[ $(module_loaded cmake/3.26.3-xi6h36u) ]] || module load cmake/3.26.3-xi6h36u
[[ $(module_loaded eigen/3.4.0-uycckhi) ]] || module load eigen/3.4.0-uycckhi
[[ $(module_loaded zlib/1.2.13-jv5y5e7) ]] || module load zlib/1.2.13-jv5y5e7
[[ $(module_loaded zstd/1.5.5-zokfqsc) ]] || module load zstd/1.5.5-zokfqsc
[[ $(module_loaded boost/1.80.0-harukoy) ]] || module load boost/1.80.0-harukoy
[[ $(module_loaded opencv) ]] || module load opencv

# Build step
mkdir outputs
mkdir -p build && cd build
cmake ..
make -j

cd ..
echo "Build completed successfully."
echo "Run the application config/eth3d_delivery_area.yaml"
./bin/main_VO --config_file config/eth3d_delivery_area.yaml