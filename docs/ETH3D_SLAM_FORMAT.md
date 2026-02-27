# ETH3D SLAM Benchmark Format Support

## Overview

The `ETH3DSLAMIterator` class provides support for the ETH3D SLAM benchmark stereo dataset format, which differs from the original ETH3D multi-view stereo format.

## Dataset Structure

The SLAM benchmark format expects the following structure:

```
dataset_name/
├── rgb/
│   ├── <timestamp1>.png
│   ├── <timestamp2>.png
│   └── ...
├── rgb2/
│   ├── <timestamp1>.png
│   ├── <timestamp2>.png
│   └── ...
├── calibration.txt           # Intrinsics for camera 1 (fx fy cx cy)
├── calibration2.txt          # Intrinsics for camera 2
├── extrinsics_1_2.txt        # 3x4 transformation matrix between cameras
├── groundtruth.txt           # Camera trajectory (optional)
└── rgb.txt                   # List of images with timestamps
```

## File Formats

### rgb.txt
Lists all images with their timestamps:
```
# timestamp filename
1234567890.123456 rgb/1234567890.123456.png
1234567890.223456 rgb/1234567890.223456.png
...
```

### groundtruth.txt
Camera trajectory in TUM format (optional):
```
# timestamp tx ty tz qx qy qz qw
1234567890.123456 0.0 0.0 0.0 0.0 0.0 0.0 1.0
...
```
- `timestamp`: Image timestamp in seconds
- `tx ty tz`: Translation vector
- `qx qy qz qw`: Rotation quaternion (x, y, z, w)

## Usage

### C++ Code

```cpp
#include "Stereo_Iterator.h"

// Create iterator for ETH3D SLAM dataset
auto iterator = Iterators::createETH3DSLAMIterator("/path/to/dataset");

// Iterate through stereo frames
StereoFrame frame;
while (iterator->hasNext())
{
    if (iterator->getNext(frame))
    {
        // Access images
        cv::Mat left = frame.left_image;
        cv::Mat right = frame.right_image;
        
        // Access ground truth (if available)
        Eigen::Matrix3d R = frame.gt_rotation;
        Eigen::Vector3d T = frame.gt_translation;
        
        // Access timestamp
        double timestamp = frame.timestamp;
    }
}
```

### In Dataset Configuration

You can integrate this into your dataset loading code by checking the dataset type:

```cpp
if (dataset_type == "ETH3D_SLAM")
{
    stereo_iterator = Iterators::createETH3DSLAMIterator(dataset_path);
}
else if (dataset_type == "ETH3D")
{
    stereo_iterator = Iterators::createETH3DIterator(stereo_pairs_path);
}
```

## Key Features

1. **Automatic Ground Truth Alignment**: If `groundtruth.txt` exists, poses are automatically aligned to image timestamps using nearest-neighbor matching.

2. **Error Handling**: The iterator gracefully handles missing ground truth and continues without it.

3. **Timestamp-based Stereo Matching**: Ensures left and right images are properly paired based on timestamps.

## Differences from Original ETH3D Format

| Feature | Original ETH3D | SLAM Benchmark |
|---------|---------------|----------------|
| Structure | Nested folders per stereo pair | Flat rgb/ and rgb2/ folders |
| Image naming | im0.png, im1.png | Timestamped PNG files |
| Ground truth | images.txt per pair | Single groundtruth.txt |
| Calibration | Embedded in dataset | Separate calibration files |

## Notes

- Images are loaded in grayscale by default
- The iterator assumes corresponding images in `rgb/` and `rgb2/` have the same filename
- Ground truth poses are optional; system continues without them
- All timestamps are in seconds (Unix time with fractional seconds)
