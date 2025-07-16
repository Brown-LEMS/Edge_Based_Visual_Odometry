# Edge-Based Visual Odometry 
## Research @ LEMS, Brown University
This repository is an internal research project at LEMS lab, Brown University. It is still under development. 

## Dependencies
* Eigen 3.X or beyond
* OpenCV 4.X (with contrib is optional)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp) (used to parse data from the dataset config file)
* glog (Optional)

## Build and Compile
Follow the standard build and compilation, namely, 
```bash
mkdir build && cd build
cmake ..
make -j
```
You shall find executive files under the ``bin`` folder. The code to be executed is ``./main_VO`` while ``./test_functions`` executes ``test_functions.cpp`` used to test certain functionalities.

### Running on Brown CCV Oscars server
A shell script ``load_modules.sh`` has been provided which enables a simple module loading process followed by build, compile, and execute. Running that shell script should be sufficient to run the entire code. <br />
Alternatively, to manually load modules, the followings are necessary: <br />
```bash
cmake/3.2XX
eigen/3.4XX
boost/1.80XX
```
where ``XX`` is the ommited version number. OpenCV (with opencv_contrib) and YAML-CPP are not existing modules, and thus they need to be installed. You will thus have to manually add their library paths to ``OpenCV_DIR`` and ``yaml-cpp_LIB``, respectively, _e.g._, ``/path-to/opencv_install/libr64/cmake/opencv4`` and ``/path-to/yaml-cpp/bin/lib64/cmake/yaml-cpp``. You will also need to edit the linkings defined in the ``CMakeLists.txt``.

## Usage
Under the ``bin`` folder, if glog is used
```bash
./main_VO
```
If not, do
```bash
./main_VO --config_file ../your-config-file.yaml
```
where the configuration file ``your-config-file.yaml`` can be customized depending on the dataset in use. See ``.yaml`` files under the ``config`` folder for more information. <br />
If no ``--config_file`` input argument is given, the default value will be used which is defined in ``cmd/main_VO.cpp`` if glog is used. Otherwise, a help message will show up. <br />
You shall find executive files under the ``bin`` folder. The code to be executed is ``./main_VO`` while ``./test_functions`` executes ``test_functions.cpp`` used to test certain functionalities. <br />

## Outputs
Output files will be written under the ``output`` folder. This part is to be updated.

## Credits
The third-order edge detection is borrowed from [Third-Order-Edge-Detector](https://github.com/C-H-Chien/Third-Order-Edge-Detector).

## Authors
Jue Han (jue_han@brown.edu) <br />
Saul Lopez Lucas (saul_lopez_lucas@brown.edu) <br />
Chiang-Heng Chien (chiang-heng_chien@brown.edu)