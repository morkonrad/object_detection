#include "common.h"
#include "clDriver.h"
#include "assert.h"
#include <stdlib.h>

constexpr auto transpose = R"(

kernel void transpose(global const float * in,
                             global float * out,
                             int w,int h)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	out[(x * h) + y] = in[(y * w) + x];
}
)";
using namespace coopcl;

static void ref_gold(
	//virtual_device& device, 
    //clTask& task,
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls,
	const std::vector<float>& input,
	std::vector<float>& output,
	const int width, const int height)
{
	/*auto d_input = device.alloc(input);
	auto d_output = device.alloc(output);
	//execute only on CPU offload=>0
	const auto ok = device.execute(
        task, 0,
		{ gs[0],gs[1],gs[2] },
		{ ls[0],ls[1],ls[2] },
		d_input, d_output,
		width, height);
	assert(ok==0);
	const auto ptr = (const float*)d_output->data();
	for(size_t i = 0;i<output.size();i++)
		output[i] = ptr[i];

	return;*/
	const auto count_gx = gs[0] / ls[0];
	const auto count_gy = gs[1] / ls[1];
	const auto count_gz = gs[2] / ls[2];

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
							const int x = gx*ls[0] + lx;
							const int y = gy*ls[1] + ly;							
							output[(x * height) + y] = input[(y * width) + x];
						}
					}
				}
			}
		}
	}
}

int main()
{	
	auto step = 0.25f;
	int err = 0;

	virtual_device device;	

    std::vector<int> widths{ 16,32,64,128,256,512,1024,2048 };
	//std::iota(widths.begin(), widths.end(), 1);
	
    std::vector<int> heights{ 16,32,64,128,256,512,1024,2048 };
	//std::iota(heights.begin(), heights.end(), 1);

	std::vector<int> depths(1);
	std::iota(depths.begin(), depths.end(), 1);

	std::vector<float> offloads;
	generate_offload_range(offloads, step);	

    constexpr auto iter = 2;
	coopcl::clTask task;
	device.build_task(task, transpose, "transpose");

	for (int testid = 0; testid < 1; testid++)
	{
		std::cout << "Testing ..." << std::endl;

		for (const auto depth : depths) {
			for (const auto height : heights) {
				for (const auto width : widths)
				{
					std::vector<float> h_input(width*height); init_random(h_input);
					std::vector<float> h_output(width*height, 0);
					std::vector<float> h_output_ref(width*height, 0);					

					auto d_input = device.alloc(h_input, true);

					ref_gold(/*device,task,*/
					{ (size_t)width,(size_t)height,1 }, 
					{ 16,16,1 }, 
					h_input, h_output_ref,
					width,height);										

					std::cout << "------------------------------" << std::endl;
					for (const auto offload : offloads)
					{
						long duration_execute = 0;
						for (int i = 0; i < iter; i++)
						{
							
							auto d_output = device.alloc(h_output);

							auto start = std::chrono::system_clock::now();							

							err = device.execute(
                                task,  offload,
								{ (size_t)width,(size_t)height,1 },
								{ 16,16,1 },
								d_input, d_output,
								width, height);

							on_error(err);

							auto b = (const float*)d_output->data();

							auto end = std::chrono::system_clock::now();
							const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
							//std::cout << "Elapsed time:\t" << et << " ms" << std::endl;
							duration_execute += et;

							assert(d_output->items() == h_output.size());
							for (auto id = 0; id < h_output_ref.size(); id++)
							{
								if (!cmpf(b[id], h_output_ref[id]))
								{
									std::cerr << "######### Something wrong at pos:\t" << id << std::endl;
									std::cerr << b[id] << " != " << h_output_ref[id] << std::endl;
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
