#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include "../include/definitions.h"
#include "../include/Dataset.h"
#include "../include/Pipeline.h"
#include "../include/Temporal_Matches.h"

#if USE_GLOGS
#include <glog/logging.h>
#endif

//> usage: (Under the bin file) ./main_VO --config_file ../config/eth3d_delivery_area.yaml

//> Define default values for the input argument
#if USE_GLOGS
DEFINE_string(config_file, "../config/tum.yaml", "config file path");
#endif

int main(int argc, char **argv)
{

	//> Get input arguments
#if USE_GLOGS
	google::ParseCommandLineFlags(&argc, &argv, true);
#else
	//> Get input argument
	--argc;
	++argv;
	std::string arg;
	int argIndx = 0, argTotal = 4;
	std::string FLAGS_config_file;

	if (argc)
	{
		arg = std::string(*argv);
		if (arg == "-h" || arg == "--help")
		{
			LOG_PRINT_HELP_MESSAGE;
			return 0;
		}
		else if (argc <= argTotal)
		{
			while (argIndx <= argTotal - 1)
			{
				if (arg == "-c" || arg == "--config_file")
				{
					argv++;
					arg = std::string(*argv);
					FLAGS_config_file = arg;
					argIndx += 2;
					break;
				}
				else
				{
					LOG_ERROR("Invalid input arguments! Follow the instruction:");
					LOG_PRINT_HELP_MESSAGE;
					return 0;
				}
				argv++;
			}
		}
		else if (argc > argTotal)
		{
			LOG_ERROR("Too many input arguments! Follow the instruction:");
			LOG_PRINT_HELP_MESSAGE;
			return 0;
		}
	}
	else
	{
		LOG_PRINT_HELP_MESSAGE;
		return 0;
	}
#endif
	YAML::Node config_map;

	try
	{
		config_map = YAML::LoadFile(FLAGS_config_file);
#if SHOW_YAML_FILE_DATA
		std::cout << config_map << std::endl;
#endif
	}
	catch (const std::exception &e)
	{
		std::cerr << "Exception: " << e.what() << std::endl;
		std::cerr << "File does not exist!" << std::endl;
	}

	//> Initialize the dataset and the pipeline
	Dataset::Ptr dataset_ = std::make_shared<Dataset>(config_map);
	Pipeline::Ptr edge_vo_system = Pipeline::Ptr(new Pipeline(dataset_));

	//> Loop over all stereo image pairs
	size_t frame_idx = 0;
    while (dataset_->stereo_iterator->hasNext())
    {
		if (!dataset_->stereo_iterator->getNext(edge_vo_system->current_frame))
        {
            std::cout << "No more image pairs to process" << std::endl;
            break;
        }

		edge_vo_system->set_Stereo_Frame_Index(frame_idx);
		edge_vo_system->Add_Stereo_Frame();
		frame_idx++;

		if (frame_idx == 2)
			break;
	}

	// Temporal_Matches temporal_matches(config_map);
	// temporal_matches.PerformEdgeBasedVO();
	return 0;
}