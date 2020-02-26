#include "common.h"
#include "clDriver.h"
#include "assert.h"
#include <stdlib.h>

constexpr auto sgemm = R"(kernel void sgemm(
                              const int M,
                              const int N,
                              const int K,
                              const __global float* A,
                              const __global float* B,
                              __global float* C)
                              {                              
								  // Thread identifiers
								  const int x = get_global_id(0); // Row ID of C (0..M)
								  const int y = get_global_id(1); // Col ID of C (0..N)

								  // Compute a single element (loop over K)
								  float acc = 0.0f;

								  for (int k=0; k<K; k++) 
									acc += A[k*M + x] * B[y*K + k];                             
								 
								  // Store the result								  
								  C[y*M + x] = acc;
                              })";

using namespace coopcl;

static void ref_gold(
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls,
	const int M, const  int N, const  int K,
	const std::vector<float>& A,
	const std::vector<float>& B,
	std::vector<float>& C)
{
	const auto count_gx = gs[0] / ls[0];
	const auto count_gy = gs[1] / ls[1];
	const auto count_gz = gs[2] / ls[2];
	
	//std::cout << "calculate gold/reference..." << std::endl;
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
							const auto globalRow = gy*ls[1] + ly;
							const auto globalCol = gx*ls[0] + lx;

							// Compute a single element (loop over K)
							float acc = 0.0f;
							for (int k = 0; k < K; k++) {
								acc += A[k*M + globalRow] * B[globalCol*K + k];
							}
							// Store the result
							C[globalCol*M + globalRow] = acc;
						}
					}
				}
			}
		}
	}
}

int main()
{	    
    auto step = 0.1f;
	int err = 0;

	virtual_device device;	

    std::vector<int> widths{32,64,128,256,512,1024};
    //std::vector<int> widths(width);
    //std::iota(widths.begin(), widths.end(), 1);

    std::vector<int> heights{32,64,128,256,512,1024};
    //std::vector<int> heights(height);
    //std::iota(heights.begin(), heights.end(), 1);

	std::vector<int> depths(1);
	std::iota(depths.begin(), depths.end(), 1);

	std::vector<float> offloads;
	generate_offload_range(offloads, step);

    constexpr auto iter = 2;

	for (int testid = 0; testid < 1; testid++)
	{
		std::cout << "Testing ..." << std::endl;

		for (const auto depth : depths) {
			for (const auto height : heights) {
				for (const auto width : widths)
				{
					const int M = width;
					const int N = M;
                    const int K = height;

                    std::vector<float> va(M*K); init_random(va);
                    std::vector<float> vb(K*N); init_random(vb);
					std::vector<float> vc(M*N, 0);
					std::vector<float> vc_ref(M*N, 0);

					auto A = device.alloc(va, true);
					auto B = device.alloc(vb, true);
					
					
					long seq_wall_time = 0;
					for (int i = 0; i < 1; i++)
					{
						auto start = std::chrono::system_clock::now();

						ref_gold({ (size_t)M,(size_t)N,1 }, { 1,1,1 }, M, N, K, va, vb, vc_ref);

						auto end = std::chrono::system_clock::now();
						const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
						seq_wall_time += et;						
					}
					//std::cout << "Sequential:\t" << seq_wall_time/iter << " ms\n";

					coopcl::clTask task;
					device.build_task(task,{ (size_t)(M),(size_t)(N),(size_t)(1) }, sgemm, "sgemm");

					std::cout << "------------------------------" << std::endl;
					for (const auto offload : offloads)
					{
						std::cout << "Offload:\t" << offload << "\n";
						long duration_execute = 0;
						for (int i = 0; i < iter; i++)
						{
							auto C = device.alloc(vc);

							auto start = std::chrono::system_clock::now();							

							err = device.execute(task,
                                offload,
								{ (size_t)M,(size_t)N,1 },
								{ (size_t)16,(size_t)16,1 },
								M, N, K, A, B, C);

							on_error(err);

							auto b = (const float*)C->data();

							auto end = std::chrono::system_clock::now();
							const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
							//std::cout << "Elapsed time:\t" << et << " ms" << std::endl;
							duration_execute += et;

                            assert(C->items() == vc_ref.size());
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
