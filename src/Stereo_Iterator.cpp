#include "Stereo_Iterator.h"

// =====================================================================================================================
// class Stereo_Iterator: image iterator to load images from dataset once at a time
//
// ChangeLogs
//    Jue  25-06-14    Initially created.
//
//> (c) LEMS, Brown University
//> Jue Han (jhan192@brown.edu)
// ======================================================================================================================

/////////////////////////
// EuRoCIterator
/////////////////////////

EuRoCIterator::EuRoCIterator(const std::string &csv_path,
                             const std::string &left_path,
                             const std::string &right_path)
    : left_path(left_path), right_path(right_path), csv_path(csv_path)
{
    csv_file.open(csv_path);
    if (!csv_file.is_open())
    {
        std::cerr << "ERROR: Could not open: " << csv_path << std::endl;
    }
}

bool EuRoCIterator::hasNext()
{
    return csv_file && csv_file.peek() != EOF;
}

void EuRoCIterator::reset()
{
    // Close and reopen the file
    csv_file.close();
    csv_file.open(csv_path);
    first_line_skipped = false;
}

bool EuRoCIterator::getNext(StereoFrame &frame)
{
    std::string line;
    while (std::getline(csv_file, line))
    {
        if (!first_line_skipped)
        {
            first_line_skipped = true;
            continue;
        }

        std::istringstream line_stream(line);
        std::string ts_str;
        std::getline(line_stream, ts_str, ',');

        double timestamp = std::stod(ts_str);
        std::string left_img = left_path + ts_str + ".png";
        std::string right_img = right_path + ts_str + ".png";

        cv::Mat left = cv::imread(left_img, cv::IMREAD_GRAYSCALE);
        cv::Mat right = cv::imread(right_img, cv::IMREAD_GRAYSCALE);

        if (!left.empty() && !right.empty())
        {
            frame.left_image = left;
            frame.right_image = right;
            frame.timestamp = timestamp;
            // Ground truth will be handled by AlignedStereoIterator
            return true;
        }

        std::cerr << "Skipping image pair: " << ts_str << std::endl;
    }

    return false;
}

/////////////////////////
// ETH3DIterator
/////////////////////////

ETH3DIterator::ETH3DIterator(const std::string &stereo_pairs_path)
{
    for (const auto &entry : std::filesystem::directory_iterator(stereo_pairs_path))
    {
        if (entry.is_directory())
        {
            folders.push_back(entry.path().string());
        }
    }
    std::sort(folders.begin(), folders.end());
}

void ETH3DIterator::reset()
{
    current_index = 0;
}

bool ETH3DIterator::hasNext()
{
    return current_index < folders.size();
}

bool ETH3DIterator::getNext(StereoFrame &frame)
{
    while (hasNext())
    {
        const std::string &folder = folders[current_index++];
        std::string left_path = folder + "/im0.png";
        std::string right_path = folder + "/im1.png";
        std::string camera_motion = folder + "/images.txt";
        cv::Mat left = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
        cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);

        // std::cout << "Left image path = " << left_path << std::endl;
        // std::cout << "Right image path = " << right_path << std::endl;

        if (!left.empty() && !right.empty())
        {
            frame.left_image = left;
            frame.right_image = right;
            frame.timestamp = static_cast<double>(current_index - 1);

            if (!readETH3DGroundTruth(camera_motion, frame))
            {
                std::cerr << "Warning: Could not read ground truth from: " << camera_motion << std::endl;
            }

            return true;
        }

        std::cerr << "Skipping bad folder: " << folder << std::endl;
    }

    return false;
}

bool ETH3DIterator::readETH3DGroundTruth(const std::string &images_file, StereoFrame &frame)
{
    std::ifstream file(images_file);
    if (!file.is_open())
    {
        return false;
    }

    std::string line;
    bool found_im0 = false;

    while (std::getline(file, line))
    {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;

        while (iss >> token)
        {
            tokens.push_back(token);
        }

        // Check if this line has the expected format and is for im0.png
        if (tokens.size() >= 10 && tokens[9] == "im0.png")
        {
            try
            {
                double qw = std::stod(tokens[1]);
                double qx = std::stod(tokens[2]);
                double qy = std::stod(tokens[3]);
                double qz = std::stod(tokens[4]);

                double tx = std::stod(tokens[5]);
                double ty = std::stod(tokens[6]);
                double tz = std::stod(tokens[7]);

                Eigen::Quaterniond q(qw, qx, qy, qz);
                Camera_Pose gt_camera_pose(q, Eigen::Vector3d(tx, ty, tz));
                frame.gt_camera_pose = gt_camera_pose;

                found_im0 = true;
                break;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error parsing ground truth: " << e.what() << std::endl;
                return false;
            }
        }
    }

    return found_im0;
}

/////////////////////////
// ETH3DSLAMIterator
/////////////////////////

ETH3DSLAMIterator::ETH3DSLAMIterator(const std::string &dataset_path)
    : dataset_path(dataset_path)
{
    if (!loadImageList())
    {
        std::cerr << "ERROR: Failed to load image list from: " << dataset_path << "/rgb.txt" << std::endl;
    }

    if (!loadGroundTruth())
    {
        std::cerr << "WARNING: Failed to load ground truth from: " << dataset_path << "/groundtruth.txt" << std::endl;
        std::cerr << "Continuing without ground truth data." << std::endl;
    }
}

bool ETH3DSLAMIterator::loadImageList()
{
    std::string rgb_file = dataset_path + "/rgb.txt";
    std::ifstream file(rgb_file);

    if (!file.is_open())
    {
        std::cerr << "ERROR: Could not open: " << rgb_file << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        double timestamp;
        std::string filename;

        if (iss >> timestamp >> filename)
        {
            image_list.push_back({timestamp, filename});
        }
    }

    std::cout << "Loaded " << image_list.size() << " stereo image pairs from rgb.txt" << std::endl;
    return !image_list.empty();
}

bool ETH3DSLAMIterator::loadGroundTruth()
{
    std::string gt_file = dataset_path + "/groundtruth.txt";
    std::ifstream file(gt_file);

    if (!file.is_open())
    {
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        double timestamp, tx, ty, tz, qx, qy, qz, qw;

        // Format: timestamp tx ty tz qx qy qz qw
        if (iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
        {
            Eigen::Quaterniond q(qw, qx, qy, qz);
            Eigen::Matrix3d R = q.toRotationMatrix();
            Eigen::Vector3d T(tx, ty, tz);

            gt_poses.emplace_back(timestamp, R, T);
        }
    }

    // Sort by timestamp for efficient lookup
    std::sort(gt_poses.begin(), gt_poses.end());

    std::cout << "Loaded " << gt_poses.size() << " ground truth poses" << std::endl;
    return !gt_poses.empty();
}

bool ETH3DSLAMIterator::findClosestGTPose(double timestamp, Eigen::Matrix3d &R, Eigen::Vector3d &T)
{
    if (gt_poses.empty())
        return false;

    // Find closest timestamp using binary search
    auto it = std::lower_bound(gt_poses.begin(), gt_poses.end(),
                               GTPose(timestamp, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()));

    if (it == gt_poses.end())
    {
        // Use last pose
        --it;
    }
    else if (it != gt_poses.begin())
    {
        // Check if previous pose is closer
        auto prev = it - 1;
        if (std::abs(prev->timestamp - timestamp) < std::abs(it->timestamp - timestamp))
        {
            it = prev;
        }
    }

    R = it->rotation;
    T = it->translation;

    return true;
}

void ETH3DSLAMIterator::reset()
{
    current_index = 0;
}

bool ETH3DSLAMIterator::hasNext()
{
    return current_index < image_list.size();
}

bool ETH3DSLAMIterator::getNext(StereoFrame &frame)
{
    if (!hasNext())
        return false;

    double timestamp = image_list[current_index].first;
    std::string filename = image_list[current_index].second;
    current_index++;

    // Construct paths for left and right images
    std::string left_path = dataset_path + "/" + filename;
    std::string right_path = dataset_path + "/rgb2/" + filename.substr(4); // Remove "rgb/" prefix and add "rgb2/"

    // Load images
    cv::Mat left = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);

    if (left.empty() || right.empty())
    {
        std::cerr << "ERROR: Failed to load images:" << std::endl;
        std::cerr << "  Left:  " << left_path << " (" << (left.empty() ? "failed" : "ok") << ")" << std::endl;
        std::cerr << "  Right: " << right_path << " (" << (right.empty() ? "failed" : "ok") << ")" << std::endl;
        return false;
    }

    frame.left_image = left;
    frame.right_image = right;
    frame.timestamp = timestamp;

    // Load ground truth if available
    if (!findClosestGTPose(timestamp, frame.gt_rotation, frame.gt_translation))
    {
        // If no ground truth, set to identity
        frame.gt_rotation = Eigen::Matrix3d::Identity();
        frame.gt_translation = Eigen::Vector3d::Zero();
    }

    return true;
}

// =============================================================
// EuRoCGTPoseIterator Implementation
// =============================================================

EuRoCGTPoseIterator::EuRoCGTPoseIterator(const std::string &gt_file,
                                         const Eigen::Matrix3d &R_frame2body,
                                         const Eigen::Vector3d &T_frame2body_vec)
{
    gt_stream.open(gt_file);
    if (!gt_stream.is_open())
    {
        std::cerr << "ERROR: Could not open ground truth file: " << gt_file << std::endl;
        return;
    }

    Eigen::Matrix4d T_frame2body = Eigen::Matrix4d::Identity();
    T_frame2body.block<3, 3>(0, 0) = R_frame2body;
    T_frame2body.block<3, 1>(0, 3) = T_frame2body_vec;

    inv_T_frame2body = T_frame2body.inverse();
}

bool EuRoCGTPoseIterator::hasNext()
{
    return gt_stream && gt_stream.peek() != EOF;
}

bool EuRoCGTPoseIterator::getNext(Eigen::Matrix3d &R_out, Eigen::Vector3d &T_out, double &timestamp)
{
    std::string line;
    while (std::getline(gt_stream, line))
    {
        if (!first_line_skipped)
        {
            first_line_skipped = true;
            continue;
        }

        std::stringstream ss(line);
        std::string val;
        std::vector<double> row;

        while (std::getline(ss, val, ','))
        {
            try
            {
                row.push_back(std::stod(val));
            }
            catch (const std::invalid_argument &e)
            {
                std::cerr << "Invalid argument: " << e.what()
                          << " for value (" << val << ") in GT file" << std::endl;
            }
            catch (const std::out_of_range &e)
            {
                std::cerr << "Out of range: " << e.what()
                          << " for value: " << val << " in GT file" << std::endl;
            }
        }

        if (row.size() < 8)
            continue;

        timestamp = row[0];
        Eigen::Vector3d T_body(row[1], row[2], row[3]);
        Eigen::Quaterniond q(row[4], row[5], row[6], row[7]);
        Eigen::Matrix3d R_body = q.toRotationMatrix();

        Eigen::Matrix4d T_world_from_body = Eigen::Matrix4d::Identity();
        T_world_from_body.block<3, 3>(0, 0) = R_body;
        T_world_from_body.block<3, 1>(0, 3) = T_body;

        Eigen::Matrix4d T_world_from_frame = (inv_T_frame2body * T_world_from_body.inverse()).inverse();

        R_out = T_world_from_frame.block<3, 3>(0, 0);
        T_out = T_world_from_frame.block<3, 1>(0, 3);
        return true;
    }

    return false;
}

// =============================================================
// GTPoseAligner Implementation
// Preload all ground truth poses from the iterator and store them in a sorted vector
// =============================================================

GTPoseAligner::GTPoseAligner(std::unique_ptr<GTPoseIterator> gt_iterator)
{
    // Preload all GT data
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    double timestamp;

    while (gt_iterator->hasNext())
    {
        if (gt_iterator->getNext(R, T, timestamp))
        {
            GTPose pose(timestamp, R, T);
            poses.emplace_back(pose);
        }
    }

    std::sort(poses.begin(), poses.end());
}

// get the aligned ground truth pose for the given image timestamp
// it takes the img_timestamp, and make changes to inputed E and T
// returns true if the closet GT pose is found, false otherwise (no GT poses available)
bool GTPoseAligner::getAlignedGT(double img_timestamp, Eigen::Matrix3d &R, Eigen::Vector3d &T)
{
    if (poses.empty())
    {
        return false;
    }

    auto it = std::lower_bound(poses.begin(), poses.end(),
                               GTPose(img_timestamp, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()));

    size_t index;
    if (it == poses.end())
    {
        // If beyond all timestamps, use the last one
        index = poses.size() - 1;
    }
    else if (it == poses.begin())
    {
        // If before all timestamps, use the first one
        index = 0;
    }
    else
    {
        // Choose between the closest timestamps before and after
        auto before_it = it - 1;

        if (std::abs(it->timestamp - img_timestamp) <
            std::abs(before_it->timestamp - img_timestamp))
        {
            index = std::distance(poses.begin(), it);
        }
        else
        {
            index = std::distance(poses.begin(), before_it);
        }
    }

    R = poses[index].rotation;
    T = poses[index].translation;
    return true;
}

// =============================================================
// AlignedStereoIterator Implementation
// =============================================================

AlignedStereoIterator::AlignedStereoIterator(
    std::unique_ptr<StereoIterator> image_iterator,
    std::unique_ptr<GTPoseAligner> gt_aligner) : image_iterator(std::move(image_iterator)),
                                                 gt_aligner(std::move(gt_aligner)) {}

bool AlignedStereoIterator::hasNext()
{
    return image_iterator->hasNext();
}

bool AlignedStereoIterator::getNext(StereoFrame &frame)
{
    if (!image_iterator->getNext(frame))
    {
        return false;
    }

    // Try to align with GT
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    if (gt_aligner->getAlignedGT(frame.timestamp, R, T))
    {
        Camera_Pose gt_camera_pose(R, T);
        frame.gt_camera_pose = gt_camera_pose;
    }

    return true;
}

void AlignedStereoIterator::reset()
{
    image_iterator->reset();
}

// =============================================================
// Iterators Implementation
// =============================================================

namespace Iterators
{
    std::unique_ptr<StereoIterator> createEuRoCIterator(
        const std::string &csv_path,
        const std::string &left_path,
        const std::string &right_path)
    {
        return std::make_unique<EuRoCIterator>(csv_path, left_path, right_path);
    }

    std::unique_ptr<StereoIterator> createETH3DIterator(
        const std::string &stereo_pairs_path)
    {
        return std::make_unique<ETH3DIterator>(stereo_pairs_path);
    }

    std::unique_ptr<StereoIterator> createETH3DSLAMIterator(
        const std::string &dataset_path)
    {
        return std::make_unique<ETH3DSLAMIterator>(dataset_path);
    }

    std::unique_ptr<GTPoseIterator> createEuRoCGTPoseIterator(
        const std::string &gt_file,
        const Eigen::Matrix3d &R_frame2body,
        const Eigen::Vector3d &T_frame2body)
    {
        return std::make_unique<EuRoCGTPoseIterator>(gt_file, R_frame2body, T_frame2body);
    }

    std::unique_ptr<StereoIterator> createAlignedEuRoCIterator(
        const std::string &csv_path,
        const std::string &left_path,
        const std::string &right_path,
        const std::string &gt_file,
        const Eigen::Matrix3d &R_frame2body,
        const Eigen::Vector3d &T_frame2body)
    {
        auto image_iterator = createEuRoCIterator(csv_path, left_path, right_path);
        auto gt_iterator = createEuRoCGTPoseIterator(gt_file, R_frame2body, T_frame2body);
        auto gt_aligner = std::make_unique<GTPoseAligner>(std::move(gt_iterator));

        return std::make_unique<AlignedStereoIterator>(
            std::move(image_iterator),
            std::move(gt_aligner));
    }
}