#include "common.h"
#include "clDriver.h"

#include "assert.h"
#include <cstdlib>


const auto k1d = R"( 
kernel void k1d(const global float* in, global float* out)
{
	const int tidx = get_global_id(0);
	out[tidx] = in[tidx];
})";

const auto k2d = R"( 
kernel void k2d(const global float* in, global float* out)
{
	const int tidx = get_global_id(0);
	const int tidy = get_global_id(1);
	out[tidx] = in[tidx];
})";

const auto k3d = R"( 
kernel void k3d(const global float* in, global float* out)
{
	const int tidx = get_global_id(0);
	const int tidy = get_global_id(1);
	const int tidz = get_global_id(2);
	out[tidx] = in[tidx];
})";

static int call_test(
	const std::string& kernel_file,
	const std::string& kernel_name,
	const std::string& jit_flags,
	const float offload,
	const std::array<size_t,3> gs,
	const std::array<size_t, 3> ls )
{
	coopcl::virtual_device device;
		
	coopcl::clTask task;
    auto err = device.build_task(task, kernel_file, kernel_name, jit_flags);
	if (err != 0)return err;

	const auto items = gs[0] * gs[1] * gs[2];
	const auto init_val = 1.1f;
	
	auto in = device.alloc<float>(items,init_val, true);
	auto out = device.alloc(items,0.0f);

	err = device.execute(task, offload, gs, ls, in, out);
	if (err != 0)return err;

    for (size_t i = 0; i < items; i++)
	{
		const auto val = out->at<float>(i);
		const auto abs_diff = std::fabs(val - init_val);
		if (abs_diff > 0.001)
		{
			std::cerr << "Some error, fixme! " << out->at<float>(i)<< std::endl;
			return -1;
		}
	}
	return 0;
}

static int test_two_or_more_kernels(const float offload,
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls)
{
	std::stringstream kernel_file;// kernel_file contains two or more kernels
	kernel_file << "struct dummy{ int d1;\n int d2;\n};\n";
	kernel_file << k1d << "\n" << k2d << "\n" << k3d;
	return call_test(kernel_file.str(), "k1d","", offload, gs, ls);	
}

static int test_one_kernel(const float offload,
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls)
{
	std::stringstream kernel_file; // kernel_file contains only one kernel
	kernel_file << "struct dummy{ int d1;\n int d2;\n};\n";
	kernel_file << k1d ;
	return call_test(kernel_file.str(), "k1d", "", offload, gs, ls);
}

static int test_no_kernels(const float offload,
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls)
{
	std::stringstream kernel_file; // kernel_file is empty
	return call_test(kernel_file.str(), "k1d", "", offload, gs, ls);
}

int main()
{
	int err = 0;
	const float offload = 0.3;
	const std::array<size_t, 3> gs = { 128,1,1 };
	const std::array<size_t, 3> ls = { 16,1,1 };

	// Kernel_file cases:
	//---------------------------------------------------
	// kernel_file case 1:  without any kernel function
	err = test_no_kernels(offload, gs, ls);
	if (err == 0)return -1; //return negative value, because expect err !=0

	// kernel_file case 2:  with only single kernel function
	err = test_one_kernel(offload, gs, ls);
	if (err != 0)return err;
	
	// kernel_file case 3:  with two or more kernel functions
	err = test_two_or_more_kernels(offload, gs, ls);
	if (err != 0)return err;
	

	// Case_2 : separate name,body,flags for CPU and GPU

	/*
	coopcl::clTask taskv2;
	std::string cpu_body, cpu_fname, cpu_jit;
	std::string gpu_body, gpu_fname, gpu_jit;	
	err = device.build_tasks(taskv2,
		cpu_body, cpu_fname, cpu_jit,
		gpu_body, gpu_fname, gpu_jit);
	if (err != 0)return err;
	*/

	std::cout << "Passed! exit..." << std::endl;
	return 0;
}
