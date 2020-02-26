#include "common.h"
#include "clDriver.h"
#include "assert.h"
#include <stdlib.h>

constexpr auto sobel_filter = R"(
__kernel void sobel_filter(
const __global uchar* inputImage, 
__global uchar* outputImage)
{
	const uint x = get_global_id(0);
    const uint y = get_global_id(1);

	const uint width = get_global_size(0);
	const uint height = get_global_size(1);

	float Gx = (float)(0);
	float Gy = Gx;
	
	const int c = x + y * width;

	if( x >= 1 && x < (width-1) && y >= 1 && y < height - 1)
	{
		float i00 = convert_float(inputImage[c - 1 - width]);
		float i10 = convert_float(inputImage[c - width]);
		float i20 = convert_float(inputImage[c + 1 - width]);
		float i01 = convert_float(inputImage[c - 1]);
		float i11 = convert_float(inputImage[c]);
		float i21 = convert_float(inputImage[c + 1]);
		float i02 = convert_float(inputImage[c - 1 + width]);
		float i12 = convert_float(inputImage[c + width]);
		float i22 = convert_float(inputImage[c + 1 + width]);

		Gx =   i00 + (float)(2) * i10 + i20 - i02  - (float)(2) * i12 - i22;
		Gy =   i00 - i20  + (float)(2)*i01 - (float)(2)*i21 + i02  -  i22;		
		
		outputImage[c] = convert_uchar(hypot(Gx, Gy)/(float)(2));
	}			
}
)";

static void ref_gold(
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls,
	std::vector<cl_uchar>& inputImage, 
	std::vector<cl_uchar>& outputImage)
{
	const auto count_gx = gs[0] / ls[0];
	const auto count_gy = gs[1] / ls[1];
	const auto count_gz = gs[2] / ls[2];
	
	const int width = gs[0];
	const int height = gs[1];

	std::cout << "calculate gold/reference..." << std::endl;
	for (auto gz = 0; gz < count_gz; gz++)
	{
		#pragma omp parallel for
		for (auto gy = 0; gy < count_gy; gy++)
		{
			for (auto gx = 0; gx < count_gx; gx++)
			{
				for (auto lz = 0; lz < ls[2]; lz++)
				{
					for (auto ly = 0; ly < ls[1]; ly++)
					{
						for (auto lx = 0; lx < ls[0]; lx++)
						{
							//kernel_func							
							int x = gx*ls[0] + lx;
							int y = gy*ls[1] + ly;							

							float Gx = 0;
							float Gy = Gx;
							
							int c = x + y * width;							
							
							if( x >= 1 && x < (width-1) && y >= 1 && y < height - 1)
							{								
								auto i00 = (float)(inputImage[c - 1 - width]);
								auto i10 = (float)(inputImage[c - width]);
								auto i20 = (float)(inputImage[c + 1 - width]);
								auto i01 = (float)(inputImage[c - 1]);
								auto i11 = (float)(inputImage[c]);
								auto i21 = (float)(inputImage[c + 1]);
								auto i02 = (float)(inputImage[c - 1 + width]);
								auto i12 = (float)(inputImage[c + width]);
								auto i22 = (float)(inputImage[c + 1 + width]);

								Gx = i00 + 2.0*i10 + i20 - i02 - 2.0*i12 - i22;
								Gy =   i00-i20+2.0*i01-2.0*i21 + i02-i22;
								
								outputImage[c] = (cl_uchar)(hypot(Gx, Gy)/2.0);

							}
						}
					}
				}
			}
		}
	}
}

using namespace coopcl;

int main()
{		
	const auto step = 0.25f;
	int err = 0;

	coopcl::virtual_device device;

	std::vector<int> widths{ 16,32,64,128,256,512 };
	//std::vector<int> widths(width);
	//std::iota(widths.begin(), widths.end(), 1);

	std::vector<int> heights{ 16, 32, 64, 128, 256, 512 };
	//std::vector<int> heights(height);
	//std::iota(heights.begin(), heights.end(), 1);

	std::vector<int> depths(1);
	std::iota(depths.begin(), depths.end(), 1);

	std::vector<float> offloads;
	generate_offload_range(offloads, step);	

    constexpr auto iter = 2;
	coopcl::clTask task;
	device.build_task(task, sobel_filter, "sobel_filter");

	for (int testid = 0; testid < 1; testid++)
	{
		std::cout << "Testing ..." << std::endl;

		for (const auto depth : depths) {
			for (const auto height : heights) {
				for (const auto width : widths)
				{
					const int M = width;
					const int N = height;

					std::vector<cl_uchar> va(width*height,1);// init_random(va);
					std::vector<cl_uchar> vb(width*height, 0);
					std::vector<cl_uchar> vb_ref(width*height, 0);					
					
					auto A = device.alloc(va,true);			
					
					//only single channel for host/reference 
					ref_gold({ (size_t)width,(size_t)height,1 }, { 1,1,1 }, va, vb_ref);

					std::cout << "------------------------------" << std::endl;
					for (const auto offload : offloads)
					{
						long duration_execute = 0;
						for (int i = 0; i < iter; i++)
						{														
							auto B = device.alloc(vb, false);

							auto start = std::chrono::system_clock::now();							
							
							err = device.execute(
                                task, offload,
								{ (size_t)M,(size_t)N,1 },
								{ (size_t)(1),1,1 },
								A,B);

							on_error(err);

							auto b = (const cl_uchar*)B->data();

							auto end = std::chrono::system_clock::now();
							const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
							//std::cout << "Elapsed time:\t" << et << " ms" << std::endl;
							duration_execute += et;

							for (auto id = 0; id < vb_ref.size(); id++)
							{
								if (b[id] != vb_ref[id])
								{
									std::cerr << "######### Something wrong at pos:\t" << id << std::endl;
									std::cerr << (int)b[id] << " != " << (int)vb_ref[id] << std::endl;
									return -1;
								}
							}
							if (iter - 1 == i)
								std::cout << "OK\t" << offload << "\t" << width << "," << height << "\n";
						}
						if (iter > 1)
							std::cout << "\t Duration avg:\t" << duration_execute / iter << " ms\n";
					}
				}
			}
		}
	}
	std::cout << "Passed! exit..." << std::endl;

	return 0;
}
