#include "common.h"
#include "clDriver.h"
#include "assert.h"
#include <stdlib.h>

constexpr auto add_3d = R"(
                        kernel void add_3d(
                        const global float* A,
                        const global float* B,
                        global float C[],
                        const float d,
                        const int width,
                        const int height)
                        {
                            const int x = get_global_id(0);
                            const int y = get_global_id(1);
                            const int z = get_global_id(2);

                            const int tid = (width * height * z) + (width * y) + x;                            

                            C[tid] = A[tid]+B[tid]+d;
                        }
                        )";


static void ref_gold(
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls,	
	const std::vector<float>& A,
	const std::vector<float>& B,
    std::vector<float>& C,
    const float d,
    const int width, const  int height)
{
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
							//kernel_func							
                            const auto z = gz*ls[2] + lz;
                            const auto y = gy*ls[1] + ly;
                            const auto x = gx*ls[0] + lx;

                            const int tid = (width * height * z) + (width * y) + x;                           

                            C[tid] = A[tid]+B[tid]+d;
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
    auto step = 0.1f;
    int err = 0;
    
	virtual_device device;    
	
	std::vector<int> widths{ 32,64,128,256,512 };
    //std::vector<int> widths(width);
    //std::iota(widths.begin(), widths.end(), 1);
	
	std::vector<int> heights{ 32,64,128,256,512 };
    //std::vector<int> heights(height);
    //std::iota(heights.begin(), heights.end(), 1);
	
	std::vector<int> depths{ 2,4,6 };
    //std::vector<int> depths(depth);
    //std::iota(depths.begin(), depths.end(), 1);

	std::vector<float> offloads;
    generate_offload_range(offloads, step);

    constexpr auto iter = 2;

    const auto d = 0.33f;

	for (int testid = 0; testid < 1; testid++)
	{
		std::cout << "Testing ..." << std::endl;

		for (const auto depth : depths) {
			for (const auto height : heights) {
				for (const auto width : widths)
				{
                    const size_t items = width*height*depth;
                    std::vector<float> va(items); init_random(va);
                    std::vector<float> vb(items); init_random(vb);                    
					std::vector<float> vc(items, 0);
					std::vector<float> vc_ref(items, 0);
								
					auto A = device.alloc(va, true);		//App+GPU here!			
					auto B = device.alloc(vb, true);
					
                    ref_gold(
                    {(size_t)width,(size_t)height,(size_t)depth },
                    { 1,1,1 },
                    va, vb, vc_ref, d, width, height);

					coopcl::clTask task;  
					device.build_task(task, add_3d, "add_3d");

					std::cout << "------------------------------" << std::endl;
					for (const auto offload : offloads)
					{
						std::cout << "Offload:\t" << offload << "\n";
						long duration_execute = 0;
						for (int i = 0; i < iter; i++)
						{							
							auto C = device.alloc(vc, false);

							auto start = std::chrono::system_clock::now();

							err = device.execute(task,
								offload,
								{ (size_t)width,(size_t)height,(size_t)depth },
								{ 16,16,1 },
								A, B, C, d, width, height);

							on_error(err);

							auto b = (const float*)C->data();

							auto end = std::chrono::system_clock::now();
							const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
							//std::cout << "Elapsed time:\t" << et << " ms" << std::endl;
							duration_execute += et;

							assert(C->items() == vc.size());
							for (auto id = 0; id < vc_ref.size(); id++)
							{
								if (!cmpf(b[id], vc_ref[id]))
								{
									std::cerr << "######### Something wrong at pos:\t" << id << std::endl;
									std::cerr << b[id] << " != " << vc_ref[id] << std::endl;
									return -1;
								}
							}
							if (iter - 1 == i)
								std::cout << "OK\t" << offload << "\t" << width << "," << height << "," << depth << "\n";
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
