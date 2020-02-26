#include "common.h"
#include "clDriver.h"
#include "assert.h"
#include <stdlib.h>

const float DAM = 0.85f;
const float numPages = 10.0f;

constexpr auto PageRank = R"(
constant float DAM=0.85f;
constant float numPages=10.0f;
__kernel void PageRank(
__global const int* rowstr,
__global const float* ranks,
__global const int* indices,
__global float* newRanks)
{
	const int id = get_global_id(0);
	float newRank = 0.0;
	for (int k = rowstr[id]; k < rowstr[id+1]; k++)
 		newRank += ranks[indices[k]];
 
	newRank = (1.0f-DAM) / numPages + DAM * newRank;
	newRanks[id] = newRank / numPages;
}
)";


static void ref_gold(
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls,
	std::vector<int>& rowstr,
	std::vector<float>& ranks,
	std::vector<int>& indices,
	std::vector<float>& newRanks)
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
							const int id = gx*ls[0]+lx;
							float newRank = 0.0;
							for (int k = rowstr[id]; k < rowstr[id + 1]; k++)
								newRank += ranks[indices[k]];

							newRank = (1.0f-DAM) / numPages + (DAM * newRank);
							newRanks[id] = newRank / numPages;
							
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
	auto width = 128;	
	auto step = 0.15f;
	int err = 0;

    virtual_device device;	
	
	std::vector<int> widths(width);
	std::iota(widths.begin(), widths.end(), 1);

	std::vector<int> heights(1);
	std::iota(heights.begin(), heights.end(), 1);

	std::vector<int> depths(1);
	std::iota(depths.begin(), depths.end(), 1);

	std::vector<float> offloads;
	generate_offload_range(offloads, step);

    constexpr auto iter = 2;

	for (int testid = 0; testid < 3; testid++)
	{
		std::cout << "Testing ..." << std::endl;

		for (const auto depth : depths) {
			for (const auto height : heights) {
				for (const auto width : widths)
				{
					const auto wwidth = width + 2;//at least 2 items
					std::vector<int> h_rowstr(wwidth); init_random(h_rowstr, 1, wwidth);
					std::vector<float> h_ranks(wwidth,1.0f/wwidth);
					std::vector<int> h_indices(wwidth); init_random(h_indices,0,wwidth-1);
					std::vector<float> h_newRanks(wwidth,0);				
					std::vector<float> h_newRanks_ref(wwidth, 0);


					auto d_rowstr=device.alloc(h_rowstr,true);
					auto d_ranks = device.alloc(h_ranks,true);
					auto d_indices = device.alloc(h_indices,true);
					

					ref_gold(
					{ (size_t)width,1,1 }, 
					{ 1,1,1 },
					h_rowstr,
					h_ranks,
					h_indices,h_newRanks_ref);

					coopcl::clTask task;
					device.build_task(task, PageRank, "PageRank");

					std::cout << "------------------------------" << std::endl;
					for (const auto offload : offloads)
					{
						long duration_execute = 0;
						for (int i = 0; i < iter; i++)
						{
							auto d_newRanks = device.alloc(h_newRanks);

							auto start = std::chrono::system_clock::now();							

							err = device.execute(
                                task, offload,
								{ (size_t)width,1,1 },
								{ 0,0,0 },
								d_rowstr,
								d_ranks,
								d_indices,
								d_newRanks);

							on_error(err);

							auto b = (const float*)d_newRanks->data();

							auto end = std::chrono::system_clock::now();
							const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
							//std::cout << "Elapsed time:\t" << et << " ms" << std::endl;
							duration_execute += et;

							assert(d_newRanks->items() == h_newRanks_ref.size());
							for (auto id = 0; id < h_newRanks_ref.size(); id++)
							{
								if (!cmpf(b[id], h_newRanks_ref[id]))
								{
									std::cerr << "######### Something wrong at pos:\t" << id << std::endl;
									std::cerr << b[id] << " != " << h_newRanks_ref[id] << std::endl;
									return -1;
								}
							}
							if (iter - 1 == i)
								std::cout << "OK\t" << offload << "\t" << width << "," << height <<"\n";
						}
						if (iter > 1)
							std::cout << "\t Duration avg:\t" << duration_execute / iter << " ms\n" ;
					}
				}
			}
		}
	}
	std::cout << "Passed! exit..." << std::endl;

	return 0;
}
