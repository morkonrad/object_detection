#pragma once
#include <vector>
#include <utility>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "surf_constants.h"

namespace ccl_surf
{
	//! Ipoint structure holds a interest point descriptor
	typedef struct {
		float x;
		float y;
		float scale;
		float orientation;
		int laplacian;
		int clusterIndex;
		float descriptor[64];
	} Ipoint;

	typedef std::vector<Ipoint> IpVec;

	bool initialize_surf_driver(		
		const std::vector<std::pair<std::string, std::string>>& kernel_path_name,
		const bool dump = false,
		const float offload = 0.0f,
		const std::string jit_flags="",
		const std::chrono::time_point<std::chrono::system_clock>* app_begin = { nullptr });

	IpVec run(const cv::Mat& img, bool use_images_in_kernelfunc = false);

	bool write_task_times_to_file(const std::string file);
}