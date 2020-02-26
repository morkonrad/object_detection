#pragma once
#include <string>
#include <ostream>

namespace iargs
{
	struct app_args
	{
		float offload = 0.0f;
		std::string app_mode{},
			img_path_a{},
			img_path_b{};

		static bool usage(std::ostream& stream)
		{
			stream << "Usage exmaple: app.exe -h" << std::endl;
			stream << "Usage exmaple: app.exe -i image_path<string> -f offload<float>" << std::endl;
			stream << "Usage exmaple: app.exe -v video_path<string> -f offload<float>" << std::endl;
			stream << "Usage exmaple: app.exe -c -f offload<float> (camera_IO mode, try use sensor)" << std::endl;
			stream << "Usage exmaple: app.exe -m img_path_a<string> img_path_b<string> -f offload<float>" << std::endl;
			return false;
		}

		void dump(std::ostream& stream)const
		{
			stream << "App_mode:\t" << app_mode << std::endl;
			stream << "Path_img_a:\t" << img_path_a << std::endl;
			stream << "Path_img_b:\t" << img_path_b << std::endl;
			stream << "Offload:\t" << offload << std::endl;
		}

		bool parse(int argc, char** argv)
		{
            bool contains_h_mode = false;
            bool contains_i_mode = false;
            bool contains_v_mode = false;
            bool contains_c_mode = false;
            bool contains_m_mode = false;
			bool contains_offload = false;

			for (int i = 0; i < argc; i++)
			{
				const std::string str = argv[i];
				if (str == "-i")
				{
					contains_i_mode = true;
					app_mode = str;
					img_path_a = argv[i + 1];
				}
				else if (str == "-h")
				{
					contains_h_mode = true;
					app_mode = str;
				}
				else if (str == "-v")
				{
					contains_v_mode = true;
					app_mode = str;
					img_path_a = argv[i + 1];
				}
				else if (str == "-c")
				{
					contains_c_mode = true;
					app_mode = str;
				}
				else if (str == "-m")
				{
					contains_m_mode = true;
					app_mode = str;
					img_path_a = argv[i + 1];
					img_path_b = argv[i + 2];

				}
				else if (str == "-f") {
					contains_offload = true;
					offload = std::atof(argv[i + 1]);
				}
			}
			
			if (app_mode == "-h") { return true; } //help_only requested!

			if (!contains_offload)
				return false;

			//check values
			if (app_mode.empty())
				return false;
			
			if (offload != -1.0f)
			{
				if (offload < 0.0f)
					return false;
				else if (offload > 1.0f)
					return false;
			}

			if (app_mode == "-i" || app_mode == "-v")
			{
				if (img_path_a.empty())
					return false;
			}
			else if (app_mode == "-m")
			{
				if (img_path_a.empty() || img_path_b.empty())
					return false;
			}

			return true;
		}
	};
}
