#include "common.h"
#include "clDriver.h"
#include "assert.h"
#include <stdlib.h>

constexpr auto atomics = R"(

kernel void atomics_add(global const int* in,
                        global int* out)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int old = atom_add(&out[0], 1);	
}
)";

using namespace coopcl;

static void ref_gold(	
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls,
	const std::vector<int>& input,
	std::vector<int>& output)
{	
	const auto count_gx = gs[0] / ls[0];
	const auto count_gy = gs[1] / ls[1];
	const auto count_gz = gs[2] / ls[2];

	std::cout << "calculate gold/reference..." << std::endl;
	for (auto gz = 0; gz < count_gz; gz++)
	{		
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
							output[0] = output[0] + 1;							
						}
					}
				}
			}
		}
	}
}

int main()
{
	int err = 0;
	virtual_device device;

	std::vector<float> offloads;
	generate_offload_range(offloads, 0.1f);

	constexpr auto iter = 1000;
	const int elements = 87;

	for (int testid = 0; testid < 1; testid++)
	{
		std::cout << "Testing ..." << std::endl;

		std::vector<int> h_input(elements);
		std::vector<int> h_output(elements, 0);
		std::vector<int> h_output_ref(elements, 0);

		auto d_input = device.alloc(h_input, true);

		coopcl::clTask task;
		device.build_task(task,atomics, "atomics_add");

		ref_gold({ (size_t)elements,(size_t)1,1 },
		{ 1,1,1 },h_input, h_output_ref);

		std::cout << "------------------------------" << std::endl;
		for (const auto offload : offloads)
		{
			long duration_execute = 0;
			for (int i = 0; i < iter; i++)
			{
				auto d_output = device.alloc(h_output);
				auto b = (const int*)d_output->data();
				auto start = std::chrono::system_clock::now();

				err = device.execute(task, offload,{ (size_t)elements,(size_t)1,1 },{ 1,1,1 },
					d_input, d_output);
				on_error(err);

				b = (const int*)d_output->data();

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
					std::cout << "OK\t" << offload << "\t" << "\n";
			}
			if (iter > 1)
				std::cout << "\t Duration avg:\t" << duration_execute / iter << " ms\n";
		}
	}
	std::cout << "Passed! exit..." << std::endl;

	return 0;
}
