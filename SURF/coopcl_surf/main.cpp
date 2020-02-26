#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "parse_args.h"
#include "kpmatch.h"
#include "ccl_surf.h"

std::chrono::time_point<std::chrono::system_clock> _app_start_point;

using namespace ccl_surf;

namespace surf_utils
{
	static void drawIPoints(cv::Mat& img, std::vector<Ipoint> &ipts)
	{
		std::cout << "Found features:\t" << ipts.size() << std::endl;
		Ipoint *ipt;
		float s, o;
		int r1, c1, r2, c2, lap;

		for (unsigned int i = 0; i < ipts.size(); i++)
		{
			ipt = &ipts.at(i);
			s = ((9.0f / 1.2f) * ipt->scale) / 3.0f;
			o = ipt->orientation;
			lap = ipt->laplacian;
			r1 = std::round(ipt->y);
			c1 = std::round(ipt->x);
			c2 = std::round(s * cos(o)) + c1;
			r2 = std::round(s * sin(o)) + r1;

			if (o) // Green line indicates orientation
				cv::line(img, cv::Point(c1, r1), cv::Point(c2, r2), cv::Scalar(0, 255, 0));
			else  // Green dot if using upright version
				cv::circle(img, cv::Point(c1, r1), 1, cv::Scalar(0, 255, 0), -1);

			if (lap == 1)
			{ // Blue circles indicate light blobs on dark backgrounds
				cv::circle(img, cv::Point(c1, r1), std::round(s), cv::Scalar(255, 0, 0), 1);
			}
			else
			{ // Red circles indicate light blobs on dark backgrounds
				cv::circle(img, cv::Point(c1, r1), std::round(s), cv::Scalar(0, 0, 255), 1);
			}

		}

	}
}

static int plot_features(const cv::Mat& frame,size_t& elapsed_time, const float offload = 0.0f, const bool draw_feats = true)
{
    int err = 0;
    cv::Mat img;
    frame.copyTo(img);    
    const auto start = std::chrono::system_clock::now();
    
	//Run
	auto ipts = ccl_surf::run(img);
    
    const auto end = std::chrono::system_clock::now();
    const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	elapsed_time = et;
    //std::cout << "Elapsed time:\t" << et << " ms\n";
    
    if (draw_feats)
    {
        // Draw the descriptors on the image
        surf_utils::drawIPoints(img, ipts);
        //show
        cv::imshow("SURF results", img);
    }
    return err;
}

static int call_stream_demo(const std::string input_file = "", const float offload = 0.0f)
{
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    cv::VideoCapture vid_capture(input_file);

    if (input_file.empty()) vid_capture = cv::VideoCapture(0);

    // Check if camera/IO opened successfully
    if (!vid_capture.isOpened()) {
        std::cerr << "Error opening video stream or file:\t" << input_file << std::endl;
        return -1;
    }
    
    cv::Mat frame;
    vid_capture >> frame;
	
	size_t elapsed_time = 0;
	size_t acc_elapsed_time = 0;
	size_t cnt_frames = 0;	
	const auto max_frames = 100;

    while (cnt_frames<max_frames)
    {        
        // Capture frame-by-frame
        vid_capture >> frame;

        // If the frame is empty, break immediately
        if (frame.empty()) break;

        plot_features(frame, elapsed_time, offload);
		acc_elapsed_time += elapsed_time;

        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(1);
        if (c == 27) break;
		cnt_frames++;        
    }
	std::cout << "Average elapsed_time:\t" << acc_elapsed_time / cnt_frames <<" ms"<< std::endl;
	std::cout << "Overall time_ms:\t" << acc_elapsed_time << std::endl;
    // When everything done, release the video capture object
    vid_capture.release();

    // Closes all the frames
    cv::destroyAllWindows();  
    
    //const auto path = "C:/Users/morkon/out.csv";
    //ccl_surf::write_task_times_to_file(path);

    return 0;
}

static int call_single_img_demo(const std::string& img_path, const float offload = 0.0f, const bool color_img = true)
{
    const auto frame = color_img == true ? cv::imread(img_path) : cv::imread(img_path, cv::IMREAD_GRAYSCALE);        
      	
	size_t elapsed_time = 0;
    const auto err = plot_features(frame,elapsed_time, offload);
    if (err != 0)return err;
    // Press  ESC on keyboard to exit
    cv::waitKey();
    return err;
}

static int call_feature_match_demo(const std::string& img_object_path,
    const std::string& img_scene_path, const float offload = 0.0f, const bool color_img = false)
{
    auto frame_object = color_img == true ? cv::imread(img_object_path) : cv::imread(img_object_path, cv::IMREAD_GRAYSCALE);
    auto frame_scene = color_img == true ? cv::imread(img_scene_path) : cv::imread(img_scene_path, cv::IMREAD_GRAYSCALE);

    const int octaves = OCTAVES;
    const int intervals = INTERVALS;
    const int sample_step = SAMPLE_STEP;
    const float threshold = THRES;
    const unsigned int initialIpts = INIT_FEATS;

    auto copy_ipoints = [](const IpVec* src, std::vector<fmatches::Ipoint>& dst)
    {
        //copy points ( stupid because they are equal, but only in separate name spaces)
        if (src == nullptr)return -1;
        for (const auto item : *src)
        {
            fmatches::Ipoint dst_point;
            std::memcpy(&dst_point, &item, sizeof(Ipoint));
            dst.push_back(dst_point);
        }
		return 0;
    };

    auto ffeats_object = ccl_surf::run(frame_object);
    std::vector<fmatches::Ipoint> feats_object;    
    copy_ipoints(&ffeats_object, feats_object);
    std::cout << "feats_object:\t" << feats_object.size() << std::endl;

    /*const auto path = "C:/Users/morkon/out.csv";
    ccl_surf::write_task_times_to_file(path);*/
    
	auto ffeats_scene = ccl_surf::run(frame_scene);
    std::vector<fmatches::Ipoint> feats_scene;    
    copy_ipoints(&ffeats_scene, feats_scene);
    std::cout << "feats_scene:\t" << feats_scene.size() << std::endl;

    fmatches::Plot_Img_Matches_ocv(frame_object, frame_scene, feats_object, feats_scene);

    return 0;
}

static void initialize_surf_driver(const iargs::app_args& args)
{
	// Initialize 
	std::cout << "Initialize driver ..." << std::endl;
	std::cout << "Build jit-functions ..." << std::endl;

	//Read openCL files with SURF kernels
#ifdef __linux
    const std::string path = "/home/morkon/Projects/SequenceStreamDemo/SURF/coopcl_surf/CLSource/";
#else
    const std::string path = "C:/Development/SequenceStreamDemo/SURF/coopcl_surf/CLSource/";
#endif

	auto append_kernel_name_path = [](const std::string& path, const std::string& file_name)->std::string
	{
		std::string file_path = path;
		file_path.append(file_name);
		return file_path;
	};

	const std::string path_des = append_kernel_name_path(path, "createDescriptors_kernel.cl");
	const std::string path_nn = append_kernel_name_path(path, "nearestNeighbor_kernel.cl");
	const std::string path_orient = append_kernel_name_path(path, "getOrientation_kernels.cl");
	const std::string path_non_max = append_kernel_name_path(path, "nonMaxSuppression_kernel.cl");
	const std::string path_hessian = append_kernel_name_path(path, "hessianDet_kernel.cl");
	const std::string path_norm = append_kernel_name_path(path, "normalizeDescriptors_kernel.cl");
	const std::string path_intimg = append_kernel_name_path(path, "integralImage_kernels.cl");

	// build pairs: <kernel_file(absolute path), kernel_function name>
	const std::vector<std::pair<std::string, std::string>> kernels =
	{
		{ path_des,"createDescriptors" },
		{ path_orient,"getOrientationStep1" },
		{ path_orient,"getOrientationStep2" },
		{ path_hessian,"hessian_det" },
		{ path_intimg,"scan" },
		{ path_intimg,"scan4" },
		{ path_intimg,"scanImage" },
		{ path_intimg,"transpose" },
		{ path_intimg, "transposeImage" },
		{ path_non_max,"non_max_supression" },
		{ path_norm,"normalizeDescriptors" }
	};

	bool dump = false;
#ifdef _DEBUG
	dump = true;
#endif // _DEBUG

	std::stringstream jit_flags;
	jit_flags << "-cl-unsafe-math-optimizations "; //This option includes the -cl-no-signed-zeros and -cl-mad-enable options see OpenCL spec.
	jit_flags << "-cl-fast-relaxed-math ";
	if (ccl_surf::initialize_surf_driver(kernels, dump, args.offload, jit_flags.str(),&_app_start_point) == false)
	{
		std::cerr << "Some error on init_surf_driver... FIXME!!" << std::endl;
		std::exit(-1);
	}
}

int main(int argc, char** argv)
{    
    /*std::stringstream ss1;
    std::stringstream ss2;   
    for (int i = 0; i < 100; i++)
    {
        const auto val = i==0?21:(i+1)*22-1;
        ss1 <<"$Sheet1.$C$" << val <<"; ";
        ss2 <<"$Sheet1.$D$" << val <<"; ";
    }
    std::cout << ss1.str() << std::endl;
    std::cout << ss2.str() << std::endl;*/

    _app_start_point  = std::chrono::system_clock::now();
    
    iargs::app_args args;

    if (!args.parse(argc, argv))
        return args.usage(std::cerr);
    
    if (args.app_mode != "-h")
    {
        args.dump(std::cout);
		initialize_surf_driver(args);		
	}

    if (args.app_mode == "-h") //help
        args.usage(std::cout);
    else if (args.app_mode == "-i") // Open/use single image file
        return call_single_img_demo(args.img_path_a, args.offload);
    else if (args.app_mode == "-v")	// Open/use single video file 
        return call_stream_demo(args.img_path_a, args.offload);
    else if (args.app_mode == "-c") //Open/use camera_sensor
        return call_stream_demo("", args.offload);
    else if (args.app_mode == "-m") //Use two images and match their features 			
        return call_feature_match_demo(args.img_path_a, args.img_path_b, args.offload);

    return 0;
}
