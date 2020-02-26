#include <cstdio>
#include <fstream>

#include "sift.h"
#include "kpmatch.h"
#include "parse_args.h"


//#define _OBSERV_ML_DATASET_

using namespace cv;

const bool up_scale = false;

#ifdef _DEBUG
static void show_sift_images(const float offload,const cv::Mat& frame)
{

	const auto octaves = SIFT::cnt_octaves();
	const auto scales = SIFT::cnt_scales();
	for (size_t oid = 0; oid < octaves; oid++)
	{
		auto fd = SIFT::get_gray(offload, frame.data, frame.cols, frame.rows, frame.channels(), oid);
		auto dbg_f = Mat(frame.rows / std::pow(2, oid), frame.cols / std::pow(2, oid), CV_32FC1, (void*)fd);
		std::stringstream fname_gray;
		fname_gray << "Frame_gray_" << oid;
		imshow(fname_gray.str(), dbg_f); (char)waitKey();

		for (size_t sid = 0; sid < scales - 1; sid++)
		{
			auto bimg = SIFT::get_blur(offload, frame.data, frame.cols, frame.rows, frame.channels(), oid, sid);
			auto dbg_bimg = Mat(frame.rows / std::pow(2, oid), frame.cols / std::pow(2, oid), CV_32FC1, (void*)bimg);
			std::stringstream fname_blur;
			fname_blur << "Frame_blur_" << oid << "_" << sid;
			imshow(fname_blur.str(), dbg_bimg); (char)waitKey();

			auto diffb = SIFT::get_diff_blur(offload, frame.data, frame.cols, frame.rows, frame.channels(), oid, sid);
			auto dbg_diffb = Mat(frame.rows / std::pow(2, oid), frame.cols / std::pow(2, oid), CV_32FC1, (void*)diffb);
			std::stringstream fname_blur_diff;
			fname_blur_diff << "Frame_blur_diff_" << oid << "_" << sid;
			imshow(fname_blur_diff.str(), dbg_diffb); (char)waitKey();

		}
	}


}
#endif // _DEBUG
static int plot_features(const cv::Mat& frame, size_t& elapsed_time, const float offload = 0.0f)
{
	int err = 0;	

	//show_sift_images(offload, frame);

	//Process frame	
	std::vector<SIFT::sfeature> feats;
	std::vector<float> descs;
	std::vector<float> orients;	

#ifdef _OBSERV_ML_DATASET_
	std::vector<float> offloads = { 0.0f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,1.0f };
	for (auto off : offloads)
	{
		descs.clear();
		feats.clear();
		orients.clear();
		err = SIFT::extract_features(elapsed_time, off, frame.data, frame.cols, frame.rows, frame.channels(), feats, descs, orients, up_scale);
		if (err != 0)return err;
	}
#else
	err = SIFT::extract_features(elapsed_time, offload, frame.data, frame.cols, frame.rows, frame.channels(), feats, descs, orients, up_scale);
	if (err != 0)return err;	
#endif

	float fx, fy;
	fx = fy = 1.0f;	
	if (up_scale) { fx = fy = 2.0f; }	
	cv::Mat showFrame;
	cv::resize(frame, showFrame, cv::Size(frame.cols * fx, frame.rows * fy), fx, fy);

	// Plot features
	size_t id = 0;
	for (const auto& f : feats) 
	{
		const auto o = orients[id++];
		const auto r1 = std::round(f.y);
		const auto c1 = std::round(f.x);
		const auto c2 = std::round(2*f.sigma * cos(o)) + c1;
		const auto r2 = std::round(2*f.sigma * sin(o)) + r1;
		cv::line(showFrame, cv::Point(c1, r1), cv::Point(c2, r2), cv::Scalar(0, 255, 0));
		cv::circle(showFrame, cv::Point(c1, r1), std::round(2*f.sigma), cv::Scalar(255, 0, 0), 1);
	}

	std::stringstream ss_msg;
	ss_msg << "Features: <" << feats.size() << ">";
	cv::putText(showFrame, ss_msg.str(), cv::Point(10, 30), cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(20, 20, 255));

	//Display the resulting frame
	imshow("Frame", showFrame);
	return err;
}

static int call_stream_demo(const std::string& input_file="",const float offload = 0.0f)
{
	// Create a VideoCapture object and open the input file
	// If the input is the web camera, pass 0 instead of the video file name
	VideoCapture vid_capture(input_file);
	
	if (input_file.empty()) vid_capture = VideoCapture(0);
	
	// Check if camera/IO opened successfully
	if (!vid_capture.isOpened()) {
		std::cerr << "Error opening video stream or file:\t"<< input_file << std::endl;
		return -1;
	}
	
	size_t elapsed_time = 0;
	size_t acc_elapsed_time = 0;
	size_t cnt_frames = 0;
	const auto max_frames = 100;

	while (cnt_frames<max_frames)
	{
		Mat frame;
		// Capture frame-by-frame
		vid_capture >> frame;

		// If the frame is empty, break immediately
		if (frame.empty())
			break;
		
		elapsed_time = 0;		
		plot_features(frame, elapsed_time, offload);
		acc_elapsed_time += elapsed_time;

		// Press  ESC on keyboard to exit
		char c = (char)waitKey(1);
		if (c == 27) break;
		cnt_frames++;
	}
	std::cout << "Average time_ms:\t" << acc_elapsed_time / cnt_frames << std::endl;
	std::cout << "Overall time_ms:\t" << acc_elapsed_time  << std::endl;
	// When everything done, release the video capture object
	vid_capture.release();

	// Closes all the frames
	destroyAllWindows();

	const auto path = "C:/Users/morkon/Sift/";
	SIFT::store_task_exec_times(path);

	return 0;
}

static int call_single_img_demo(const std::string& img_path, const float offload = 0.0f, const bool color_img = true)
{
    const auto frame = color_img ? cv::imread(img_path) : cv::imread(img_path, IMREAD_GRAYSCALE);
		
	//const float ratio_x = 2.0f; const float ratio_y = 2.0f;
	//auto vimg = SIFT::test_ColorToGray_Resize(offload, frame.data, frame.cols, frame.rows, frame.channels(), ratio_x, ratio_y);
	//cv::Mat img( (int)(frame.rows * ratio_y), (int)(frame.cols * ratio_x), CV_32FC1,vimg.data());
	
	//auto vimg = SIFT::test_kResize(offload, frame.data, frame.cols, frame.rows, frame.channels(), ratio_x, ratio_y);
	//cv::Mat img( (int)(frame.rows * ratio_y), (int)(frame.cols * ratio_x), CV_8UC3, vimg.data());
	//cv::imshow("Resize img",img);	

	size_t elapsed_time = 0;
	const auto err = plot_features(frame,elapsed_time,offload);
	if (err!=0)return err;	

	waitKey();
	return 0;
	
}

static int call_feature_match_demo(const std::string& img_object_path, 
	const std::string& img_scene_path, const float offload = 0.0f, const bool color_img=false)
{
    auto frame_object = color_img ? cv::imread(img_object_path): cv::imread(img_object_path, IMREAD_GRAYSCALE);
    auto frame_scene = color_img ? cv::imread(img_scene_path): cv::imread(img_scene_path, IMREAD_GRAYSCALE);

	std::vector<SIFT::sfeature> feats_object,feats_scene;
	std::vector<float> descs_object,descs_scene;
	std::vector<float> orients_object,orients_scene;

	size_t elapsed_time = 0;
	int err = SIFT::extract_features(elapsed_time,offload, 
		frame_object.data, frame_object.cols, frame_object.rows,frame_object.channels(),
		feats_object, descs_object, orients_object,up_scale);
	if (err != 0)return err;
	std::cout << "feats_object:\t" << feats_object.size() << std::endl;
	
	elapsed_time = 0;
	err = SIFT::extract_features(elapsed_time, offload, 
		frame_scene.data, frame_scene.cols, frame_scene.rows, frame_scene.channels(),
		feats_scene, descs_scene, orients_scene, up_scale);
	if (err != 0)return err;
	std::cout << "feats_scene:\t" << feats_scene.size() << std::endl;

	std::vector<fmatches::sift_feat> kp_object, kp_scene;
	fmatches::ConvertToSift_feat<SIFT::sfeature>(feats_object, orients_object, descs_object, kp_object);
	fmatches::ConvertToSift_feat<SIFT::sfeature>(feats_scene, orients_scene, descs_scene, kp_scene);
	fmatches::Plot_Img_Matches_ocv(frame_object, frame_scene, kp_object, kp_scene, up_scale);
	
	return 0;
}

int main(int argc, char** argv)
{	    
    iargs::app_args args;

    if (!args.parse(argc, argv))
        return iargs::app_args::usage(std::cerr);
	
    if (args.app_mode == "-h") //help
        iargs::app_args::usage(std::cout);
    else if(args.app_mode == "-i") // Open/use single image file
        return call_single_img_demo(args.img_path_a, args.offload);
    else if (args.app_mode == "-v")	// Open/use single video file
        return call_stream_demo(args.img_path_a, args.offload);
    else if (args.app_mode == "-c") //Open/use camera_sensor
        return call_stream_demo("", args.offload);
    else if (args.app_mode == "-m") //Use two images and match their features
        return call_feature_match_demo(args.img_path_a ,args.img_path_b ,args.offload);
	
	return 0;
}
