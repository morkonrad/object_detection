#pragma once

#include <vector>

namespace SIFT
{
#ifdef _DEBUG
	
	size_t cnt_octaves();
	
	size_t cnt_scales();

	const float* get_blur(const float offload,
		const unsigned char* input_color_image,
		const int width, const int height, const int color_depth, 
		const int oid, const int sid);

	const float* get_diff_blur(const float offload,
		const unsigned char* input_color_image,
		const int width, const int height, const int color_depth,
		const int oid, const int sid);

	const float* get_gray(const float offload,
		const unsigned char* input_color_image,
		const int width, const int height, 
		const int color_depth,const int oid);

	std::vector<std::uint8_t> test_kResize(const float offload,
		const unsigned char* input_color_image,
		const int width, const int height,
		const int color_depth,
		const float ratio_x, const float ratio_y);

	std::vector<float> test_ColorToGray_Resize(const float offload,
		const unsigned char* input_color_image,
		const int width, const int height,
		const int color_depth,
		const float ratio_x, const float ratio_y);

#endif // DEBUG

	struct sfeature  
	{
		float x;
		float y;
		float sigma;
		size_t scale_id;
		size_t octave_id;
	};

	int extract_features(
		size_t& elapsed_time,
		float offload,
		const unsigned char* input_color_image,
		int width,int height,
		int color_depth,
		std::vector<sfeature>& features,
		bool up_scale = false);

	int extract_features(
		size_t& elapsed_time,
		float offload,
		const unsigned char* input_color_image,
		int width, int height,
		int color_depth,
		std::vector<sfeature>& features,
		std::vector<float>& descriptors,
		std::vector<float>& orientations,
		bool up_scale = false);

	void store_task_exec_times(const std::string& path);
}