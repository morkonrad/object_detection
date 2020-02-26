#include "clDriver.h"
#include <cassert>
#include <cstdlib>

constexpr auto taskA = R"(
                        kernel void kA(global int* A)                        
                        {
                            const int tid = get_global_id(0);                                                       
                            A[tid] = 10;
                        }
                        )";

constexpr auto taskB = R"(
                        kernel void kB(const global int* A,
									  global int* B)                        
                        {
                            const int tid = get_global_id(0);                                                       
                            B[tid] = A[tid]+1;
                        }
                        )";

constexpr auto taskC = R"(
                        kernel void kC(const global int* A,
									  global int* C)                        
                        {
                            const int tid = get_global_id(0);                                                       
                            C[tid] = A[tid]+2;
                        }
                        )";

constexpr auto taskD = R"(
                        kernel void kD(const global int* A,									  
									  global int* D)                        
                        {
                            const int tid = get_global_id(0);                                                       
                            D[tid] = A[tid]+3;
                        }
                        )";

constexpr auto taskE = R"(
                        kernel void kE(const global int* B,
									  const global int* C,
									  const global int* D,
									  global int* E)                        
                        {
                            const int tid = get_global_id(0);                                                       
                            E[tid] = B[tid]+C[tid]+D[tid];
                        }
                        )";

template<typename T>
static auto compare = [](const std::vector<T>& c1,const std::vector<T>& c2,const T val)->bool
{
    if(c1.size()!=c2.size())return false;

    bool ok=true;
    for(size_t i=0;i<c1.size();i++)
    {
        if(c1[i]!=c2[i] || c1[i]!=val || c2[i]!=val)
        {
            std::cerr <<"Wrong value: {"<<c1[i]<<","<<c2[i]<<","<< val << "} pos: " << i << std::endl;
            ok&=false;
        }
    }

    return ok;
};

static int test_dag0(const size_t items)
{
	//Simple task_graph consist of 3 tasks
	// 1 independent task A
	// 2 dependent tasks B(A),C(B)	
	/*
	<BEGIN>
	[A]
	 | 
	[B]
	 |
	[C]
	<END>
	*/

	//A = 10 
	//B(A) = 11 >> B=A+1
	//C(B) = 13 >> C=B+2		
	coopcl::virtual_device device;

	auto bA = device.alloc<int>(items,false);
	auto bB = device.alloc<int>(items, false);
	auto bC = device.alloc<int>(items, false);    

	coopcl::clTask task_A;			
	device.build_task(task_A,taskA, "kA");

	coopcl::clTask task_B;
	device.build_task(task_B, taskB, "kB");
	task_B.add_dependence(&task_A);

	coopcl::clTask task_C;	
	device.build_task(task_C, taskC, "kC");
	task_C.add_dependence(&task_B);

	std::cout << "Execute ...\n";
    
    device.execute_async(task_A, 1.0f, { items,1,1 }, { 16,1,1 }, bA);    
	device.execute_async(task_B, 0.5f, { items,1,1 }, { 16,1,1 }, bA, bB);    
	device.execute_async(task_C, 0.0f, { items,1,1 }, { 16,1,1 }, bB, bC);
	task_C.wait();

    std::cout << "Validate ..." << std::endl;
	for (size_t i = 0; i < bC->items(); i++)
	{
		const auto val = bC->at<int>(i);
		if (val != 13) {			
            std::cerr <<"Wrong value: "<< val << " pos: " << i << std::endl;
			return -1;
		}
	}

	std::cout << "Passed ok, exit!" << std::endl;
	return 0;

}

static int test_dag1(const size_t items)
{
	//Simple task_graph consist of 5 tasks
	// 3 dependent, data parallel tasks B,C,D
	// 1 independent task A	
	// 1 dependent task E
	/*
	<BEGIN>
	   [A]
	  / | \
	[B][C][D]
	  \ | /
	   [E]
	<END>
	*/
	//A = 10 
	//B(A) = 11 >> B=A+1
	//C(A) = 12 >> C=A+2
	//D(A) = 13 >> D=A+3
	//E(B,C,D) = B+C+D = 11+12+13 = 36

	coopcl::virtual_device device;
	

	auto bA = device.alloc<int>(items, false);
	auto bB = device.alloc<int>(items, false);
	auto bC = device.alloc<int>(items, false);
	auto bD = device.alloc<int>(items, false);
	auto bE = device.alloc<int>(items, false);

	coopcl::clTask task_A;
	device.build_task(task_A, taskA, "kA");

	coopcl::clTask  task_B;
	device.build_task(task_B, taskB, "kB");
	task_B.add_dependence(&task_A);

	coopcl::clTask  task_C;
	device.build_task(task_C, taskC, "kC");
	task_C.add_dependence(&task_A);

	coopcl::clTask  task_D;
	device.build_task(task_D, taskD, "kD");
	task_D.add_dependence(&task_A);

	coopcl::clTask task_E;
	device.build_task(task_E, taskE, "kE");
	task_E.add_dependence(&task_B);
	task_E.add_dependence(&task_C);
	task_E.add_dependence(&task_D);

	std::cout << "Execute ...\n";

	std::vector<int> cv, gv;

	device.execute_async(task_A, 1.0f, { items,1,1 }, { 16,1,1 }, bA);
	//bA->val(cv, gv);
	device.execute_async(task_B, 0.75f, { items,1,1 }, { 16,1,1 }, bA, bB);
	//bB->val(cv, gv);
	device.execute_async(task_C, 0.5f, { items,1,1 }, { 16,1,1 }, bA, bC);
	//bC->val(cv, gv);
	device.execute_async(task_D, 0.25f, { items,1,1 }, { 16,1,1 }, bA, bD);
	//bD->val(cv, gv);
	device.execute_async(task_E, 0.0f, { items,1,1 }, { 16,1,1 }, bB, bC, bD, bE);
	task_E.wait();

	std::cout << "Validate ..." << std::endl;
	//bE->val(cv, gv);	

	for (size_t i = 0; i < bE->items(); i++)
	{
        const auto val = bE->at<int>(i);
        if (val != 36) {
            std::cerr <<"Wrong value: "<<val << " pos: " << i << std::endl;
			return -1;
		}
	}

	std::cout << "Passed ok, exit!" << std::endl;
	return 0;
}

static int paper(const size_t items)
{
	//Simple task_graph consist of 4 tasks	
	/*
	<BEGIN>
	 [A]
	/   \
   [B]  [C]
	\   /
	 [D]
	<END>
	*/
	//A = 10 
	//B(A) = 11 >> B=A+1
	//C(A) = 12 >> C=A+2
	//D(B,C) = 23 >> D=B+C

	constexpr auto tasks = R"(
kernel void kA(global int* A)                        
{
const int tid = get_global_id(0);                                                       
A[tid] = 10;
}

kernel void kB(const global int* A,global int* B)                        
{
const int tid = get_global_id(0);                                                       
B[tid] = A[tid]+1;
}

kernel void kC(const global int* A,global int* C)                        
{
const int tid = get_global_id(0);                                                       
C[tid] = A[tid]+2;
}

kernel void kD(const global int* B,
const global int* C,global int* D)                        
{
const int tid = get_global_id(0); 
D[tid] = B[tid]+C[tid];
}
)";

	coopcl::virtual_device device;
	

	auto mA = device.alloc<int>(items);
	auto mB = device.alloc<int>(items);
	auto mC = device.alloc<int>(items);
	auto mD = device.alloc<int>(items);

	coopcl::clTask taskA;	
	device.build_task(taskA,  tasks, "kA");
	
	coopcl::clTask taskB;	
	device.build_task(taskB, tasks, "kB");
	taskB.add_dependence(&taskA);

	coopcl::clTask taskC;	
	device.build_task(taskC, tasks, "kC");
	taskC.add_dependence(&taskA);

	coopcl::clTask taskD;	
	device.build_task(taskD, tasks, "kD");
	taskD.add_dependence(&taskB);
	taskD.add_dependence(&taskC);

	const std::array<size_t, 3> ndr = { items,1,1 };
	const std::array<size_t, 3> wgs = { 16,1,1 };
	
	
	for (int i = 0;i < 10;i++) 
	{		
		device.execute_async(taskA, 0.0f, ndr, wgs, mA);
		device.execute_async(taskB, 0.8f, ndr, wgs, mA, mB);
		device.execute_async(taskC, 0.5f, ndr, wgs, mA, mC);
		device.execute_async(taskD, 1.0f, ndr, wgs, mB, mC, mD);
		taskD.wait();

	}
	
	for (int i = 0;i < items;i++)
	{
		const auto val = mD->at<int>(i);
		if (val != 23)
		{
			std::cerr << "Some error at pos i = " << i << std::endl;
			return -1;
		}
	}

	std::cout << "Passed,ok!" << std::endl;
	return 0;

}

int main()
{
	auto ok = paper(1024*1e3);
	if (ok != 0)return ok;

	ok = test_dag0(128*1e6);
    if(ok!=0)return ok;

    return test_dag1(200*1e5);
}
