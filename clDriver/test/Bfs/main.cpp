#include "common.h"
#include "clDriver.h"
#include "assert.h"
#include <stdlib.h>

typedef struct {
	int starting;
	int no_of_edges;
} Node;

constexpr auto bfs = R"(

typedef struct{
	int starting;
	int no_of_edges;
} Node;


__kernel void bfs(		const __global Node* g_graph_nodes,
						const __global int* g_graph_edges, 
						__global char* g_graph_mask, 
						__global char* g_updating_graph_mask, 
						const __global char* g_graph_visited, 
						__global int* g_cost, 
						const  int no_of_nodes)
{
	int tid = get_global_id(0);
	if( tid<no_of_nodes && g_graph_mask[tid])
	{
		g_graph_mask[tid]=false;
		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
		{
			int id = g_graph_edges[i];
			if(!g_graph_visited[id])
			{
				//g_cost[id]=g_cost[tid]+1; // bug data-race!(undef. results)
				g_cost[tid] = id + 1;
				g_updating_graph_mask[id]=true;
			}
		}
	}	
}
)";


static void ref_gold(
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls,
	const std::vector<Node>& g_graph_nodes,
	const std::vector<int>& g_graph_edges,
	std::vector<char>& g_graph_mask,
	std::vector<char>& g_updating_graph_mask,
	const std::vector<char>& g_graph_visited,
	std::vector<int>& g_cost,
	const int no_of_nodes)
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
							const int tid = gx*ls[0]+lx;
							if (tid < no_of_nodes && g_graph_mask[tid])
							{
								g_graph_mask[tid] = false;
								for (int i = g_graph_nodes[tid].starting; i < (g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
								{
									int id = g_graph_edges[i];
									if (!g_graph_visited[id])
									{
										g_cost[tid] = id + 1;
										g_updating_graph_mask[id] = true;
									}
								}
							}							
						}
					}
				}
			}
		}
	}
}

void init_nodes(std::vector<Node>& graph,const int begin,const int end)
{
	if (end < begin)return;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis_int(begin, end);
	for (auto& val : graph) {
		val.starting = dis_int(gen);
		val.no_of_edges = dis_int(gen);
		const auto sum = val.starting + val.no_of_edges;
		
		if (sum > graph.size())		
			val.no_of_edges = graph.size() - val.starting;		
	}
}
using namespace coopcl;

int main()
{	
	auto width = 256;	
	auto step = 0.25f;
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

	coopcl::clTask task;
	device.build_task(task, bfs, "bfs");

	for (int testid = 0; testid < 2; testid++)
	{
		std::cout << "Testing ..." << std::endl;

		for (const auto depth : depths) {
			for (const auto height : heights) {
				for (const auto width : widths)
				{
					const int no_of_nodes = width;
					std::vector<Node> h_graph_nodes(no_of_nodes); init_nodes(h_graph_nodes,1,h_graph_nodes.size());
					std::vector<int> h_graph_edges(no_of_nodes); init_random(h_graph_edges, 1, no_of_nodes - 1);
					
					std::vector<char> h_graph_mask(no_of_nodes,1);
					std::vector<char> h_graph_mask_ref(no_of_nodes, 1);

					std::vector<char> h_updating_graph_mask(no_of_nodes);
					std::vector<char> h_updating_graph_mask_ref(no_of_nodes);

					std::vector<char> h_graph_visited(no_of_nodes,0);
					std::vector<int> h_cost(no_of_nodes,0);
					std::vector<int> h_cost_ref(no_of_nodes, 0);

					auto d_graph_nodes=device.alloc(h_graph_nodes,true);
					auto d_graph_edges = device.alloc(h_graph_edges, true);					
					auto d_graph_visited = device.alloc(h_graph_visited,true);				


					ref_gold({ (size_t)no_of_nodes,1,1 }, { 1,1,1 },
						h_graph_nodes,h_graph_edges,
						h_graph_mask_ref, h_updating_graph_mask_ref,
						h_graph_visited,h_cost_ref,
						no_of_nodes );
					

					std::cout << "------------------------------" << std::endl;
					for (const auto offload : offloads)
					{
						long duration_execute = 0;
						for (int i = 0; i < iter; i++)
						{
							auto d_graph_mask = device.alloc(h_graph_mask);
							auto d_updating_graph_mask = device.alloc(h_updating_graph_mask);
							auto d_cost = device.alloc(h_cost);

							auto start = std::chrono::system_clock::now();
														
							err = device.execute(
                                task, offload,
								{ (size_t)no_of_nodes,1,1 }, { 0,0,0 },
								d_graph_nodes, d_graph_edges,
								d_graph_mask, d_updating_graph_mask,
								d_graph_visited, d_cost,
								no_of_nodes);

							on_error(err);

							auto b = (const int*)d_cost->data();

							auto end = std::chrono::system_clock::now();
							const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
							//std::cout << "Elapsed time:\t" << et << " ms" << std::endl;
							duration_execute += et;

							assert(d_cost->items() == h_cost_ref.size());							
							for (auto id = 0; id < h_cost_ref.size(); id++)
							{
								if (!cmpf(b[id], h_cost_ref[id]))
								{
									std::cerr << "######### Something wrong at pos:\t" << id << std::endl;
									std::cerr << b[id] << " != " << h_cost_ref[id] << std::endl;
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
