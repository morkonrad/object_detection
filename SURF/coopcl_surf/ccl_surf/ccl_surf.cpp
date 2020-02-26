#include "ccl_surf.h"
#include "clDriver.h"

#include <fstream>
#include <sstream>


namespace surf_items 
{
	//#define _sync_calls
	static std::unique_ptr<coopcl::virtual_device> device{ nullptr };
	static bool is_kernel_use_image{ false };
	const static std::vector<std::string> expected_surf_kernel_names = {
		"createDescriptors","getOrientationStep1" ,"getOrientationStep2" ,"hessian_det" ,
		"scan" ,"scan4" ,"scanImage" ,"transpose" , "transposeImage" ,"non_max_supression",
		"normalizeDescriptors" };

	static std::map<std::string, std::unique_ptr<coopcl::clTask>> surf_tasks;
	static inline bool isUsingImages() { return is_kernel_use_image; };
	static float surf_offload{ 0 };

	static unsigned int roundUp(unsigned int value, unsigned int multiple) {

		unsigned int remainder = value % multiple;

		// Make the value a multiple of multiple
		if (remainder != 0) {
			value += (multiple - remainder);
		}

		return value;
	}

}

using namespace surf_items;

class ResponseLayer
{
	int width;
	int height;
	int step;
	int filter;

	std::unique_ptr<coopcl::clMemory> d_responses;
	std::unique_ptr<coopcl::clMemory> d_laplacian;
	std::unique_ptr<coopcl::clTask> task_hessian_det;

public:
    ResponseLayer(int width, int height, int step, int filter)
	{
		this->width = width;
		this->height = height;
		this->step = step;
		this->filter = filter;

#ifndef _sync_calls

		auto& task_Hessian = *surf_tasks.at("hessian_det");
		task_hessian_det = std::make_unique<coopcl::clTask>();
#ifdef _PROFILE_CLTASK_
		task_hessian_det->set_app_begin_time_point(task_Hessian.get_app_begin_time_point());
#endif
        auto err = device->build_task(*task_hessian_det, task_Hessian.body(), task_Hessian.name(), task_Hessian.jit_flags());
		if (err != 0)throw std::runtime_error("Couldn't build task Hessian, fixme!!");

		// add dependencies 
		// each Hessian needs to wait for SURF
		task_hessian_det->add_dependence(surf_tasks.at("transpose").get());

#endif // _sync_calls

		if (isUsingImages()) 
		{
			//this->d_laplacian = cl_allocImage(height, width, 'i');
			//this->d_responses = cl_allocImage(height, width, 'f');
		}		
		else 
		{			
			d_laplacian = device->alloc<int>(width * height);
			d_responses = device->alloc<float>(width*height);
		}
	}

    int getWidth()const{return this->width;}
    int getHeight()const{return this->height;}
    int getStep()const{return this->step;}
    int getFilter()const{return this->filter;}

	void write_task_profile(std::ostream& out)const
	{
#ifdef _TASK_INSTRUMENTATION_
		if (task_hessian_det) task_hessian_det->write_task_profile(out);
#endif // _TASK_INSTRUMENTATION_
	}

    std::unique_ptr<coopcl::clTask>& getTaskHessian() { return task_hessian_det; }
    std::unique_ptr<coopcl::clMemory>& getLaplacian(){ return d_laplacian; }
    std::unique_ptr<coopcl::clMemory>& getResponses(){ return d_responses; }
};

class FastHessian 
{

private:

	// Based on the octave (row) and interval (column), this lookup table
	// identifies the appropriate determinant layer
	const int filter_map[MAX_OCTAVES][MAX_INTERVALS] = { { 0, 1, 2, 3 },
	{ 1, 3, 4, 5 },
	{ 3, 5, 6, 7 },
	{ 5, 7, 8, 9 },
	{ 7, 9,10,11 } };

	void createResponseMap(int octaves, int imgWidth, int imgHeight, int sample_step)
	{
		int w = (imgWidth / sample_step);
		int h = (imgHeight / sample_step);
		int s = (sample_step);

		// Calculate approximated determinant of hessian values
		if (octaves >= 1)
		{
			this->responseMap.push_back(std::make_unique<ResponseLayer>(w, h, s, 9));
			this->responseMap.push_back(std::make_unique < ResponseLayer>(w, h, s, 15));
			this->responseMap.push_back(std::make_unique < ResponseLayer>(w, h, s, 21));
			this->responseMap.push_back(std::make_unique < ResponseLayer>(w, h, s, 27));
		}

		if (octaves >= 2)
		{
			this->responseMap.push_back(std::make_unique < ResponseLayer>(w / 2, h / 2, s * 2, 39));
			this->responseMap.push_back(std::make_unique < ResponseLayer>(w / 2, h / 2, s * 2, 51));
		}

		if (octaves >= 3)
		{
			this->responseMap.push_back(std::make_unique < ResponseLayer>(w / 4, h / 4, s * 4, 75));
			this->responseMap.push_back(std::make_unique < ResponseLayer>(w / 4, h / 4, s * 4, 99));
		}

		if (octaves >= 4)
		{
			this->responseMap.push_back(std::make_unique < ResponseLayer>(w / 8, h / 8, s * 8, 147));
			this->responseMap.push_back(std::make_unique < ResponseLayer>(w / 8, h / 8, s * 8, 195));
		}

		if (octaves >= 5)
		{
			this->responseMap.push_back(std::make_unique < ResponseLayer>(w / 16, h / 16, s * 16, 291));
			this->responseMap.push_back(std::make_unique < ResponseLayer>(w / 16, h / 16, s * 16, 387));
		}
	}

	//! Number of Ipoints
	int num_ipts;

	//! Number of Octaves
	int octaves;

	//! Number of Intervals per octave
	int intervals;

	//! Initial sampling step for Ipoint detection
	int sample_step;

	//! Threshold value for blob responses
	float thres;

	std::vector<std::unique_ptr<ResponseLayer>> responseMap;

	//! Number of Ipoints on GPU 
	std::unique_ptr<coopcl::clMemory> d_ipt_count;

	std::vector<std::unique_ptr<coopcl::clTask>> tasks_non_max;

public:

	//-------------------------------------------------------
	//! Constructor
    FastHessian(int i_height, int i_width, const int octaves,
		const int intervals, const int sample_step, const float thres)
	{
		// Initialize variables with bounds-checked values
		this->octaves = (octaves > 0 && octaves <= 4 ? octaves : MAX_OCTAVES);
		this->intervals = (intervals > 0 && intervals <= 4 ? intervals : INTERVALS);
		this->sample_step = (sample_step > 0 && sample_step <= 6 ? sample_step : SAMPLE_STEP);
		this->thres = (thres >= 0 ? thres : THRES);
		this->num_ipts = 0;
		this->d_ipt_count = device->alloc<int>(1);

		// Create the Hessian response map objects
		createResponseMap(octaves, i_width, i_height, sample_step);

#ifndef _sync_calls
		auto& task_non_max = *surf_tasks.at("non_max_supression");
		tasks_non_max.resize(this->octaves * 2);
		for (auto& t : tasks_non_max)
		{
			t = std::make_unique<coopcl::clTask>();
#ifdef _PROFILE_CLTASK_
			t->set_app_begin_time_point(task_non_max.get_app_begin_time_point());
#endif // _PROFILE_CLTASK_
            const auto err = device->build_task(*t, task_non_max.body(), task_non_max.name(), task_non_max.jit_flags());
			if (err != 0) {
				throw std::runtime_error("Couldn't build task non_max_supression, fixme!!!");
			}
			//add dependencies 
			// each non_max_supression waits  for all Hessian
			for (auto& rm : responseMap)			
				t->add_dependence(rm->getTaskHessian().get());
		}
#endif
	}

    int computeHessianDet(std::unique_ptr<coopcl::clMemory>& d_intImage, int i_width, int i_height)
	{
		num_ipts = 0;		
		
		std::memcpy(d_ipt_count->data(), &num_ipts, sizeof(int));

		// set matrix size and x,y threads per block
		const int BLOCK_DIM = 16;
		int err = 0;
		size_t localWorkSize[2] = { BLOCK_DIM,BLOCK_DIM };
		size_t globalWorkSize[2];
#ifdef _sync_calls
		auto& task_hessian = *surf_tasks["hessian_det"];
#endif
		std::array<size_t, 3> gs, ls;

		for (unsigned int i = 0; i < this->responseMap.size(); i++)
		{
			auto& responses = responseMap.at(i)->getResponses();
			auto& laplacian = responseMap.at(i)->getLaplacian();

			int step = this->responseMap.at(i)->getStep();
			int filter = this->responseMap.at(i)->getFilter();
			int layerWidth = this->responseMap.at(i)->getWidth();
			int layerHeight = this->responseMap.at(i)->getHeight();

			globalWorkSize[0] = roundUp(layerWidth, localWorkSize[0]);
			globalWorkSize[1] = roundUp(layerHeight, localWorkSize[1]);

			gs = { globalWorkSize[0],globalWorkSize[1],1 };
			ls = { localWorkSize[0],localWorkSize[1],1 };			

#ifdef _sync_calls
			err = device->execute(task_hessian, surf_offload, gs, ls, d_intImage, i_width, i_height, responses, laplacian, layerWidth, layerHeight, step, filter);
#else
			auto& task_hessian_det = responseMap.at(i)->getTaskHessian();
			err = device->execute_async(*task_hessian_det, surf_offload, gs, ls, d_intImage, i_width, i_height, responses, laplacian, layerWidth, layerHeight, step, filter);
#endif
			if (err != 0)return err;
			
		}
//#ifndef _sync_calls
//		for (unsigned int i = 0; i < this->responseMap.size(); i++)
//			responseMap.at(i)->getTaskHessian()->wait();
//
//		return device->wait();
//#endif
		return err;
	}

    int getIpoints(const cv::Mat& img,
		std::unique_ptr<coopcl::clMemory>& d_intImage, 
		std::unique_ptr<coopcl::clMemory>& d_laplacian,
		std::unique_ptr<coopcl::clMemory>& d_pixPos, 
		std::unique_ptr<coopcl::clMemory>& d_scale, int maxIpts)
	{
		// Compute the Hessian determinants		
		auto err = computeHessianDet(d_intImage, img.cols, img.rows);
		if (err != 0)return err;

		// Determine which points are interesting
		// kernels: non_max_suppression kernel
		err = selectIpoints(d_laplacian, d_pixPos, d_scale, maxIpts);
		if (err != 0)return err;
		
		std::memcpy(&num_ipts, d_ipt_count->data(), sizeof(num_ipts));
		return num_ipts;
	}

    int selectIpoints(std::unique_ptr<coopcl::clMemory>& d_laplacian,
		std::unique_ptr<coopcl::clMemory>& d_pixPos,
		std::unique_ptr<coopcl::clMemory>& d_scale, int maxPoints)
	{

		// The search for extrema (the most interesting point in a neighborhood)
		// is done by non-maximal suppression
		const int BLOCK_W = 16;
		const int BLOCK_H = 16;
		int err = 0;		
#ifdef _sync_calls		
		auto& task_non_max = *surf_tasks["non_max_supression"];
#endif
		std::array<size_t, 3> gs, ls;

		size_t tid = 0;
		// Run the kernel for each octave
		for (int o = 0; o < octaves; o++)
		{
			for (int i = 0; i <= 1; i++)
			{
				auto& bResponse = this->responseMap.at(filter_map[o][i])->getResponses();
				int bWidth = this->responseMap.at(filter_map[o][i])->getWidth();
				int bHeight = this->responseMap.at(filter_map[o][i])->getHeight();
				int bFilter = this->responseMap.at(filter_map[o][i])->getFilter();

				auto& mResponse = this->responseMap.at(filter_map[o][i + 1])->getResponses();
				int mWidth = this->responseMap.at(filter_map[o][i + 1])->getWidth();
				int mHeight = this->responseMap.at(filter_map[o][i + 1])->getHeight();
				int mFilter = this->responseMap.at(filter_map[o][i + 1])->getFilter();
				auto& mLaplacian = this->responseMap.at(filter_map[o][i + 1])->getLaplacian();

				auto& tResponse = this->responseMap.at(filter_map[o][i + 2])->getResponses();
				int tWidth = this->responseMap.at(filter_map[o][i + 2])->getWidth();
				int tHeight = this->responseMap.at(filter_map[o][i + 2])->getHeight();
				int tFilter = this->responseMap.at(filter_map[o][i + 2])->getFilter();
				int tStep = this->responseMap.at(filter_map[o][i + 2])->getStep();

				size_t localWorkSize[2] = { BLOCK_W, BLOCK_H };
				size_t globalWorkSize[2] = { roundUp(mWidth, BLOCK_W),roundUp(mHeight, BLOCK_H) };				
				gs = { globalWorkSize[0],globalWorkSize[1],1 };
				ls = { localWorkSize[0],localWorkSize[1],1 };				
#ifdef _sync_calls
				err = device->execute(task_non_max, surf_offload, gs, ls, tResponse, tWidth, tHeight, tFilter, tStep,
					mResponse, mLaplacian, mWidth, mHeight, mFilter, bResponse, bWidth, bHeight,
					bFilter, d_ipt_count, d_pixPos, d_scale, d_laplacian, maxPoints, thres);
#else
				err = device->execute_async(*tasks_non_max[tid++], surf_offload, gs, ls, tResponse, tWidth, tHeight, tFilter, tStep,
					mResponse, mLaplacian, mWidth, mHeight, mFilter, bResponse, bWidth, bHeight,
					bFilter, d_ipt_count, d_pixPos, d_scale, d_laplacian, maxPoints, thres);
#endif
				if(err != 0)return err;				
			}
		}
#ifndef _sync_calls
		return device->wait();
#endif
		return err;
	}

	void write_task_profiles(std::ostream& out)const
	{
#ifdef _TASK_INSTRUMENTATION_
		for (auto& non_max : tasks_non_max){
			if(non_max) non_max->write_task_profile(out);
		}
		
		for (auto& hess_det : responseMap) {
			if(hess_det) hess_det->write_task_profile(out);
		}
#endif
	}

	//! Reset the state of the data
    void reset()
	{
		/*int numIpts = 0;
		cl_copyBufferToDevice(this->d_ipt_count, &numIpts, sizeof(int));*/
	}
};

class Surf 
{
	private:

	// The actual number of ipoints for this image
	int numIpts{ 0 };

	//! The amount of ipoints we have allocated space for
	int maxIpts{ 0 };

	//! A fast Hessian object that will be used for detecting ipoints
	std::unique_ptr<FastHessian> fh;

	//! The integral image
	std::unique_ptr<coopcl::clMemory> d_intImage;
	std::unique_ptr<coopcl::clMemory> d_tmpIntImage;   // orig orientation
	std::unique_ptr<coopcl::clMemory> d_tmpIntImageT1; // transposed
	std::unique_ptr<coopcl::clMemory> d_tmpIntImageT2; // transposed

	//! Number of surf descriptors
	std::unique_ptr<coopcl::clMemory> d_length;
	
	//! Array of Descriptors for each Ipoint
	std::unique_ptr<coopcl::clMemory> d_desc;
	
	//! Orientation of each Ipoint an array of float
	std::unique_ptr<coopcl::clMemory> d_orientation;
	std::unique_ptr<coopcl::clMemory> d_gauss25;
	std::unique_ptr<coopcl::clMemory> d_id;
	std::unique_ptr<coopcl::clMemory> d_i;
	std::unique_ptr<coopcl::clMemory> d_j;

	//! Position buffer on the device
	std::unique_ptr<coopcl::clMemory> d_pixPos;

	//! Scale buffer on the device
	std::unique_ptr<coopcl::clMemory> d_scale;

	//! Laplacian buffer on the device
	std::unique_ptr<coopcl::clMemory> d_laplacian;

	//! Res buffer on the device
	std::unique_ptr<coopcl::clMemory> d_res;

	const static int j[16];

	const static int i[16];

	const static unsigned int id[13];

	const static float gauss25[49];

	size_t image_rows{ 0 };
	size_t image_cols{ 0 };
	
	//! Convert image from 4 channel color to 4 channel grayscale	
	static cv::Mat getGray(const  cv::Mat& img)
	{
		// Check we have been supplied a non-null img pointer
		if (img.empty())
		{
			printf("Unable to create grayscale image.  No image supplied");
			exit(-1);
		}

		cv::Mat gray8;
		cv::Mat gray32;

		// Allocate space for the grayscale
		gray32 = cv::Mat(img.rows, img.cols, CV_32FC1, 1);

		if (img.channels() == 1) {
			gray8 = img;
		}
		else
		{
			gray8 = cv::Mat(img.rows, img.cols, CV_8UC1, 1);
			cv::cvtColor(img, gray8, cv::COLOR_BGR2GRAY);
		}

		gray8.convertTo(gray32, CV_32FC1, 1.0 / 255.0, 0);

		return gray32;
	}

    int computeIntegralImage(const cv::Mat&  source)
	{		
		// TODO This call takes  several ms (speed it up?)		
		//! convert the image to single channel 32f
        cv::Mat img = getGray(source);

		// set up variables for data access
		int height = img.rows;
		int width = img.cols;
		float *data = (float*)img.data;
		
		if (isUsingImages())
		{
			//// Copy the data to the GPU
			//cl_copyImageToDevice(this->d_intImage, data, height, width);
			//scan_kernel = this->kernel_list[KERNEL_SCANIMAGE];
			//transpose_kernel = this->kernel_list[KERNEL_TRANSPOSEIMAGE];
		}
		else
		{
			// Copy the image
			//cl_copyBufferToDevice(this->d_intImage, data, sizeof(float)*width*height);
			std::memcpy(d_intImage->data(), img.data, sizeof(float)*width*height);

			//// If it is possible to use the vector scan (scan4) use
			//// it, otherwise, use the regular scan
			//if(cl_deviceIsAMD() && width % 4 == 0 && height % 4 == 0)
			//{
			//// NOTE Change this to KERNEL_SCAN when running verification code.
			////      The reference code doesn't use a vector type and
			////      scan4 produces a slightly different integral image
			//scan_kernel = this->kernel_list[KERNEL_SCAN4];
			//}
			//else
			//{
			//scan_kernel = this->kernel_list[KERNEL_SCAN];
			//}
			//transpose_kernel = this->kernel_list[KERNEL_TRANSPOSE];
		}

		// -----------------------------------------------------------------
		// Step 1: Perform integral summation on the rows
		// -----------------------------------------------------------------
		size_t localWorkSize1[2] = { 64, 1 };
        size_t globalWorkSize1[2] = { 64, (size_t)height };
		std::array<size_t, 3> gs, ls;
		gs = { globalWorkSize1[0],globalWorkSize1[1],1 };
		ls = { localWorkSize1[0],localWorkSize1[1],1 };
		auto& scan_task = *surf_tasks.at("scan");
		scan_task.set_ndr_dim_to_divide(1);
		auto err = device->execute(scan_task, surf_offload, gs, ls, this->d_intImage, this->d_tmpIntImage, height, width);
		if (err != 0)return err;
		
		// -----------------------------------------------------------------
		// Step 2: Transpose
		// -----------------------------------------------------------------
		size_t localWorkSize2[]={16, 16};
		size_t globalWorkSize2[]={roundUp(width,16), roundUp(height,16)};		
		gs = { globalWorkSize2[0],globalWorkSize2[1],1 };
		ls = { localWorkSize2[0],localWorkSize2[1],1 };
		auto& transpose_task = *surf_tasks["transpose"];
		err = device->execute(transpose_task, surf_offload, gs, ls, this->d_tmpIntImage, this->d_tmpIntImageT1, height, width);
		if (err != 0)return err;
		

		// -----------------------------------------------------------------
		// Step 3: Run integral summation on the rows again (same as columns
		//         integral since we've transposed).
		// -----------------------------------------------------------------

		int heightT = width;
		int widthT = height;

		size_t localWorkSize3[2]={64, 1};
        size_t globalWorkSize3[2]={64, (size_t)heightT};
		gs = { globalWorkSize3[0],globalWorkSize3[1],1 };
		ls = { localWorkSize3[0],localWorkSize3[1],1 };
		err = device->execute(scan_task, surf_offload, gs, ls, this->d_tmpIntImageT1, this->d_tmpIntImageT2, heightT, widthT);
		if (err != 0)return err;
		
		// -----------------------------------------------------------------
		// Step 4: Transpose back
		// -----------------------------------------------------------------
		size_t localWorkSize4[]={16, 16};
		size_t globalWorkSize4[]={roundUp(widthT,16), roundUp(heightT,16)};		
		gs = { globalWorkSize4[0],globalWorkSize4[1],1 };
		ls = { localWorkSize4[0],localWorkSize4[1],1 };
		err = device->execute(transpose_task, surf_offload, gs, ls, this->d_tmpIntImageT2, this->d_intImage, heightT, widthT);
		if (err != 0)return err;

		return err;
	}

    int createDescriptors(int i_width, int i_height)
	{
		const size_t threadsPerWG = DESC_TH_WGS;
		const size_t wgsPerIpt = DESC_WGS_ITEM;		
		
		size_t localWorkSizeSurf64[2] = { threadsPerWG,1 };
		size_t globalWorkSizeSurf64[2] = { (wgsPerIpt*threadsPerWG),(size_t)numIpts };

		std::array<size_t, 3> gs, ls;
		auto& task_descriptor = *surf_tasks["createDescriptors"];
		task_descriptor.set_ndr_dim_to_divide(1);
		auto& task_norm = *surf_tasks["normalizeDescriptors"];

		gs = { globalWorkSizeSurf64[0],globalWorkSizeSurf64[1],1 };
		ls = { localWorkSizeSurf64[0],localWorkSizeSurf64[1],1 };
		auto err = device->execute(task_descriptor,surf_offload,gs,ls, 
			d_intImage, i_width, i_height, d_scale, d_desc, d_pixPos, d_orientation, d_length, d_j, d_i);
		if (err != 0)return err;
		
		size_t localWorkSizeNorm64[] = { DESC_SIZE };
        size_t globallWorkSizeNorm64[] = { (size_t)this->numIpts*DESC_SIZE };
		
		
		gs = { globallWorkSizeNorm64[0],1,1 };
		ls = { localWorkSizeNorm64[0],1,1 };
		err = device->execute(task_norm, surf_offload, gs, ls, d_desc, d_length);
		if (err != 0)return err;

		return err;
	}

    int getOrientations(int i_width, int i_height)
	{
		auto& task_orient_step1 = *surf_tasks["getOrientationStep1"];
		auto& task_orient_step2 = *surf_tasks["getOrientationStep2"];

		std::array<size_t, 3> gs, ls;
		size_t localWorkSize1[] = { ORIENT_WGS_STEP1 };
		size_t globalWorkSize1[] = { (size_t)this->numIpts*(size_t)ORIENT_WGS_STEP1 };
				
		gs = { globalWorkSize1[0],1,1 };
		ls = { localWorkSize1[0],1,1 };
		auto err = device->execute(task_orient_step1,surf_offload,gs,ls,
			d_intImage, d_scale, d_pixPos, d_gauss25, d_id, i_width, i_height, d_res);
		if (err != 0)return err;

		size_t localWorkSize2[] = { ORIENT_WGS_STEP2 };
		size_t globalWorkSize2[] = { (size_t)numIpts * (size_t)ORIENT_WGS_STEP2 };
		
		
		gs = { globalWorkSize2[0],1,1 };
		ls = { localWorkSize2[0],1,1 };
		err = device->execute(task_orient_step2, surf_offload, gs, ls, d_orientation, d_res);
		if (err != 0)return err;

		return err;
	}

public:
	//! Constructor
    Surf(const int initialPoints, const  int i_height, const int i_width, const int octaves,
		const int intervals, const int sample_step, const float threshold)
	{
		if (device == nullptr) device = std::make_unique<coopcl::virtual_device>();

		image_rows = i_height;
		image_cols = i_width;

		fh = std::make_unique<FastHessian>(i_height, i_width, octaves, intervals, sample_step, threshold);

		//   // Once we know the size of the image, successive frames should stay
		//   // the same size, so we can just allocate the space once for the integral
		//   // image and intermediate data
		if (isUsingImages())
		{
			//this->d_intImage = cl_allocImage(i_height, i_width, 'f');
			//this->d_tmpIntImage = cl_allocImage(i_height, i_width, 'f');
			//this->d_tmpIntImageT1 = cl_allocImage(i_width, i_height, 'f');
			//this->d_tmpIntImageT2 = cl_allocImage(i_width, i_height, 'f');
		}
		else
		{
			
			d_intImage = device->alloc<float>(image_cols*image_rows);
			d_tmpIntImage = device->alloc<float>(image_cols*image_rows);

			// These two are unnecessary for buffers, but required for images, so
			// we'll use them for buffers as well to keep the code clean

			d_tmpIntImageT1 = device->alloc<float>(image_cols*image_rows);
			d_tmpIntImageT2 = device->alloc<float>(image_cols*image_rows);

		}
		// Allocate constant data on device
		
		d_gauss25 = device->alloc<float>(49, &Surf::gauss25[0]);		
		d_id = device->alloc<unsigned int>(13, &Surf::id[0]);		
		d_i = device->alloc<int>(16, &Surf::i[0]);		
		d_j = device->alloc<int>(16, &Surf::j[0]);

		// Allocate buffers for each of the interesting points.  We don't know
		// how many there are initially, so must allocate more than enough space
		
		d_scale = device->alloc<float>(initialPoints);				
		d_pixPos = device->alloc<cl_float2>(initialPoints);		
		d_laplacian = device->alloc<int>(initialPoints);		
		
		// These buffers used to wait for the number of actual ipts to be known
		// before being allocated, instead now we'll only allocate them once
		// so that we can take advantage of optimized data transfers and reallocate
		// them if there's not enough space available

		d_length = device->alloc<float>(initialPoints * DESC_SIZE);
		d_desc = device->alloc<float>(initialPoints * DESC_SIZE);
		d_res = device->alloc<cl_float4>(initialPoints*109);
		d_orientation = device->alloc<float>(initialPoints);

		// This is how much space is available for Ipts
		this->maxIpts = initialPoints;
	}
	
    ccl_surf::IpVec retrieveDescriptors()const
	{
		ccl_surf::IpVec ipts;
		if(numIpts == 0) return ipts;		
		ipts.resize(numIpts);
		
		const float* begin_desc = static_cast<const float*>(d_desc->data());
		for(int i= 0;i<(this->numIpts);i++)
		{		
			ccl_surf::Ipoint ipt;
		    ipt.x = d_pixPos->at<cl_float2>(i).x;
		    ipt.y = d_pixPos->at<cl_float2>(i).y;
		    ipt.scale = d_scale->at<float>(i);
		    ipt.laplacian = d_laplacian->at<int>(i);
		    ipt.orientation = d_orientation->at<float>(i);			
		    memcpy(ipt.descriptor, begin_desc+(i*64), sizeof(float)*64);	
			ipts[i] = ipt;
		}

		return ipts;
	}

    int run(const cv::Mat&  img)
	{
		//Perform the scan sum of the image (populates d_intImage)
		//kernels: scan (x2), transpose (x2)
		auto err = this->computeIntegralImage(img);
		if (err != 0)return err;

		//Determines the points of interest
		//kernels: init_det, hessian_det (x12), non_max_suppression (x3)		
		this->numIpts = this->fh->getIpoints(img,
			this->d_intImage, this->d_laplacian,
			this->d_pixPos, this->d_scale, this->maxIpts);


		// Verify that there was enough space allocated for the number of Ipoints found
		if (this->numIpts >= this->maxIpts)
		{
			// If not enough space existed, we need to reallocate space and
			// run the kernels again
			printf("Not enough space for Ipoints, reallocating and running again\n");
			this->maxIpts = this->numIpts * 2;

			//this->reallocateIptBuffers();        
			this->fh->reset();

			numIpts = fh->getIpoints(img, this->d_intImage, this->d_laplacian,
				this->d_pixPos, this->d_scale, this->maxIpts);
			
		}

		//printf("There were %d interest points\n", this->numIpts);

		// Main SURF-64 loop assigns orientations and gets descriptors    
		if (this->numIpts == 0) return -2;

		// GPU kernel: getOrientation1 (1x), getOrientation2 (1x)
		err = getOrientations(img.cols, img.rows);
		if (err != 0)return err;

		// GPU kernel: surf64descriptor (1x), norm64descriptor (1x)
		err = createDescriptors(img.cols, img.rows);
		if (err != 0)return err;

		return 0;
	}
	
	size_t img_rows()const { return image_rows; }
	size_t img_cols()const { return image_cols; }

	void write_task_profiles(std::ostream& out)const
	{
		if(fh) fh->write_task_profiles(out);
	}

};

const int Surf::j[] = { -12, -7, -2, 3,
-12, -7, -2, 3,
-12, -7, -2, 3,
-12, -7, -2, 3 };

const int Surf::i[] = { -12,-12,-12,-12,
-7, -7, -7, -7,
-2, -2, -2, -2,
3,  3,  3,  3 };

const unsigned int Surf::id[] = { 6,5,4,3,2,1,0,1,2,3,4,5,6 };

const float Surf::gauss25[] = {
	0.02350693969273f, 0.01849121369071f, 0.01239503121241f, 0.00708015417522f, 0.00344628101733f, 0.00142945847484f, 0.00050524879060f,
	0.02169964028389f, 0.01706954162243f, 0.01144205592615f, 0.00653580605408f, 0.00318131834134f, 0.00131955648461f, 0.00046640341759f,
	0.01706954162243f, 0.01342737701584f, 0.00900063997939f, 0.00514124713667f, 0.00250251364222f, 0.00103799989504f, 0.00036688592278f,
	0.01144205592615f, 0.00900063997939f, 0.00603330940534f, 0.00344628101733f, 0.00167748505986f, 0.00069579213743f, 0.00024593098864f,
	0.00653580605408f, 0.00514124713667f, 0.00344628101733f, 0.00196854695367f, 0.00095819467066f, 0.00039744277546f, 0.00014047800980f,
	0.00318131834134f, 0.00250251364222f, 0.00167748505986f, 0.00095819467066f, 0.00046640341759f, 0.00019345616757f, 0.00006837798818f,
	0.00131955648461f, 0.00103799989504f, 0.00069579213743f, 0.00039744277546f, 0.00019345616757f, 0.00008024231247f, 0.00002836202103f };

namespace surf_items 
{
	static std::unique_ptr<Surf> alg_surf{ nullptr };

	static inline bool check_surf_taks_name(const std::string& task_name)
	{
		bool valid_name = 0;
		for (const auto& name : expected_surf_kernel_names)
		{
			if (name == task_name) {
				valid_name = true;
				return true;
			}
		}

		std::cerr << "Expected kernel function names:" << std::endl;
		for (const auto& name : expected_surf_kernel_names)
			std::cerr << name << std::endl;

		return false;
	}
}

namespace ccl_surf 
{
    IpVec run(const cv::Mat& img, bool use_images_in_kernelfunc)
	{
		if (device == nullptr)
		{
			std::cerr << "Please initialize_surf_driver first .. exit!" << std::endl;
			return{};
		}

		is_kernel_use_image = use_images_in_kernelfunc;

		if (alg_surf == nullptr)
		{
			alg_surf = std::make_unique<Surf>(INIT_FEATS, img.rows, img.cols,
				OCTAVES, INTERVALS, SAMPLE_STEP, THRES);
		}
		else
		{
			if ((alg_surf->img_rows() < img.rows) || (alg_surf->img_cols() < img.cols))
			{
				alg_surf = std::make_unique<Surf>(INIT_FEATS, img.rows, img.cols,
					OCTAVES, INTERVALS, SAMPLE_STEP, THRES);
			}
		}
		auto err = alg_surf->run(img);
		if (err != 0)
		{
			std::cerr << "Some err on main surf loop, FIXME!!" << std::endl;
			return{};
		}

		return alg_surf->retrieveDescriptors();
	}

    bool initialize_surf_driver(
		const std::vector<std::pair<std::string, std::string>>& kernel_path_name,
		const bool dump, const float offload, const std::string jit_flags,
		const std::chrono::time_point<std::chrono::system_clock>* app_begin)
	{
		if (device == nullptr)
			device = std::make_unique<coopcl::virtual_device>();

		surf_offload = offload;

		for (const auto& path_name : kernel_path_name)
		{
			//Open and read kernel files
			std::ifstream ifs(path_name.first);

			const std::string file_txt((std::istreambuf_iterator<char>(ifs)),
				std::istreambuf_iterator<char>());

			if (check_surf_taks_name(path_name.second) != true)return false;

			if (dump)std::cout << "Build task:\t" << path_name.second << std::endl;

			auto cltask = std::make_unique<coopcl::clTask>();
#ifdef _PROFILE_CLTASK_
			cltask->set_app_begin_time_point(app_begin);
#endif
			//create clTask
            const auto err = device->build_task(*cltask,
				file_txt, path_name.second, jit_flags);
			if (err != 0)return false;

			surf_tasks.emplace(path_name.second, std::move(cltask));
		}

		return true;
	}

	bool write_task_profiles_to_file(const std::string file)
	{
		if (file.empty())return false;
		std::ofstream ofs(file);
		if (!ofs.is_open())
		{
			std::cerr << "Couldn't open:\t" << file << " ,fixme!!!" << std::endl;
			return false;
		}

#ifdef _TASK_INSTRUMENTATION_		
		if (alg_surf)alg_surf->write_task_profiles(ofs);

		for (auto& task : surf_items::surf_tasks)			
			task.second->write_task_profile(ofs);
		
#endif // _TASK_INSTRUMENTATION_
		return true;
	}
}
