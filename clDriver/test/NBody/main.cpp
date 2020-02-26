#include "common.h"
#include "clDriver.h"
#include "assert.h"
#include <stdlib.h>

#define UNROLL_FACTOR  8

constexpr auto nbody_sim = R"(
#define UNROLL_FACTOR  8
__kernel void nbody_sim(
		const __global float4* pos, 
		const __global float4* vel,
		int numBodies ,
		float deltaTime, 
		float epsSqr,
		__global float4* newPosition, 
		__global float4* newVelocity) 
		{
			unsigned int gid = get_global_id(0);
			float4 myPos = pos[gid];
			float4 acc = (float4)0.0f;

			int i = 0;
			for (; (i+UNROLL_FACTOR) < numBodies; ) 
			{
				#pragma unroll UNROLL_FACTOR
				for(int j = 0; j < UNROLL_FACTOR; j++,i++) 
				{
					float4 p = pos[i];
					float4 r;
					r.xyz = p.xyz - myPos.xyz;
					float distSqr = r.x * r.x  +  r.y * r.y  +  r.z * r.z;

					float invDist = 1.0f / sqrt(distSqr + epsSqr);
					float invDistCube = invDist * invDist * invDist;
					float s = p.w * invDistCube;

					// accumulate effect of all particles
					acc.xyz += s * r.xyz;
				}
			}
			
			for (; i < numBodies; i++) 
			{
				float4 p = pos[i];

				float4 r;
				r.xyz = p.xyz - myPos.xyz;
				float distSqr = r.x * r.x  +  r.y * r.y  +  r.z * r.z;

				float invDist = 1.0f / sqrt(distSqr + epsSqr);
				float invDistCube = invDist * invDist * invDist;
				float s = p.w * invDistCube;

				// accumulate effect of all particles
				acc.xyz += s * r.xyz;
			}

			float4 oldVel = vel[gid];

			// updated position and velocity
			float4 newPos;
			newPos.xyz = myPos.xyz + oldVel.xyz * deltaTime + acc.xyz * 0.5f * deltaTime * deltaTime;
			newPos.w = myPos.w;

			float4 newVel;
			newVel.xyz = oldVel.xyz + acc.xyz * deltaTime;
			newVel.w = oldVel.w;

			// write to global memory
			newPosition[gid] = newPos;
			newVelocity[gid] = newVel;
	}
)";


static void ref_gold(
	const std::array<size_t, 3> gs,
	const std::array<size_t, 3> ls,
	std::vector<cl_float4>& pos,
	std::vector<cl_float4>& vel,
	int numBodies,
	float deltaTime,
	float epsSqr,
	std::vector<cl_float4>& newPosition,
	std::vector<cl_float4>& newVelocity)
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
							unsigned int gid = gx*ls[0]+lx;
							cl_float4 myPos = pos[gid];
							cl_float4 acc = { 0.0f,0.0f,0.0f,0.0f };

							int i = 0;
							for (; (i + UNROLL_FACTOR) < numBodies; )
							{
								#pragma unroll UNROLL_FACTOR
								for (int j = 0; j < UNROLL_FACTOR; j++, i++)
								{
									cl_float4 p = pos[i];
									cl_float4 r;
									//r.s[0]yz = p.s[0]yz - myPos.s[0]yz;
									r.s[0] = p.s[0] - myPos.s[0];
									r.s[1] = p.s[1] - myPos.s[1];
									r.s[2] = p.s[2] - myPos.s[2];
									
									float distSqr = r.s[0] * r.s[0] + r.s[1] * r.s[1] + r.s[2] * r.s[2];

									float invDist = 1.0f / sqrt(distSqr + epsSqr);
									float invDistCube = invDist * invDist * invDist;
									float s = p.s[3] * invDistCube;

									// accumulate effect of all particles
									//acc.s[0]yz += s * r.s[0]yz;
									acc.s[0] += s * r.s[0];
									acc.s[1] += s * r.s[1];
									acc.s[2] += s * r.s[2];
								}
							}

							for (; i < numBodies; i++)
							{
								cl_float4 p = pos[i];

								cl_float4 r;
								//r.s[0]yz = p.s[0]yz - myPos.s[0]yz;
								r.s[0] = p.s[0] - myPos.s[0];
								r.s[1] = p.s[1] - myPos.s[1];
								r.s[2] = p.s[2] - myPos.s[2];

								float distSqr = r.s[0] * r.s[0] + r.s[1] * r.s[1] + r.s[2] * r.s[2];

								float invDist = 1.0f / sqrt(distSqr + epsSqr);
								float invDistCube = invDist * invDist * invDist;
								float s = p.s[3] * invDistCube;

								// accumulate effect of all particles
								//acc.s[0]yz += s * r.s[0]yz;
								acc.s[0] += s * r.s[0];
								acc.s[1] += s * r.s[1];
								acc.s[2] += s * r.s[2];
							}

							cl_float4 oldVel = vel[gid];

							// updated position and velocity
							cl_float4 newPos;
							//newPos.s[0]yz = myPos.s[0]yz + oldVel.s[0]yz * deltaTime + acc.s[0]yz * 0.5f * deltaTime * deltaTime;
							newPos.s[0] = myPos.s[0] + oldVel.s[0] * deltaTime + acc.s[0] * 0.5f * deltaTime * deltaTime;
							newPos.s[1] = myPos.s[1] + oldVel.s[1] * deltaTime + acc.s[1] * 0.5f * deltaTime * deltaTime;
							newPos.s[2] = myPos.s[2] + oldVel.s[2] * deltaTime + acc.s[2] * 0.5f * deltaTime * deltaTime;

							newPos.s[3] = myPos.s[3];

							cl_float4 newVel;
							//newVel.s[0]yz = oldVel.s[0]yz + acc.s[0]yz * deltaTime;
							newVel.s[0] = oldVel.s[0] + acc.s[0] * deltaTime;
							newVel.s[1] = oldVel.s[1] + acc.s[1] * deltaTime;
							newVel.s[2] = oldVel.s[2] + acc.s[2] * deltaTime;

							newVel.s[3] = oldVel.s[3];

							// write to global memory
							newPosition[gid] = newPos;
							newVelocity[gid] = newVel;
						
						}
					}
				}
			}
		}
	}
}

void init_cl4(std::vector<cl_float4>& items)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis_real(0.1, 1.0);
	for (auto& val : items)
	{
		val.s[0] = dis_real(gen);
		val.s[1] = dis_real(gen);
		val.s[2] = dis_real(gen);
		val.s[3] = dis_real(gen);
	}
}

using namespace coopcl;

int main()
{		
	auto step = 0.1f;
	int err = 0;

	virtual_device device;	

	std::vector<int> widths{ 32,64,128,256,512,1024,2048,4096,8192 };
	//std::iota(widths.begin(), widths.end(), 1);

	std::vector<int> heights(1);
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
					const int numBodies = width;
					const float deltaTime = 0.01f;
					const float epsSqr = 0.535f;

					std::vector<cl_float4> h_pos(numBodies); init_cl4(h_pos);
					std::vector<cl_float4> h_vel(numBodies); init_cl4(h_vel);

					std::vector<cl_float4> h_newPosition(numBodies, { 0,0,0,0 });
					std::vector<cl_float4> h_newVelocity(numBodies, { 0,0,0,0 });

					std::vector<cl_float4> h_newPosition_ref(numBodies, { 0,0,0,0 });
					std::vector<cl_float4> h_newVelocity_ref(numBodies, { 0,0,0,0 });

					auto d_pos = device.alloc(h_pos,true);
					auto d_vel = device.alloc(h_vel, true);					

					

					ref_gold(
						{ (size_t)numBodies,1,1 }, 
						{ 1,1,1 }, 
						h_pos,
						h_vel,
						numBodies,
						deltaTime,
						epsSqr,
						h_newPosition_ref,
						h_newVelocity_ref);

					coopcl::clTask task; 
					device.build_task(task,nbody_sim, "nbody_sim");

					std::cout << "------------------------------" << std::endl;
					for (const auto offload : offloads)
					{
						long duration_execute = 0;
						for (int i = 0; i < iter; i++)
						{
							auto d_newPosition = device.alloc(h_newPosition);
							auto d_newVelocity = device.alloc(h_newVelocity);

                            auto start = std::chrono::system_clock::now();

							err = device.execute(
                                task, offload,
								{ (size_t)numBodies,1,1 },
								{ 16,1,1 },
								d_pos,
								d_vel,
								numBodies,
								deltaTime,
								epsSqr,
								d_newPosition,
								d_newVelocity);

							on_error(err);

							auto b = (const cl_float4*)d_newVelocity->data();

							auto end = std::chrono::system_clock::now();
							const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
							//std::cout << "Elapsed time:\t" << et << " ms" << std::endl;
							duration_execute += et;

							assert(d_newVelocity->items() == h_newVelocity_ref.size());
							for (auto id = 0; id < h_newVelocity_ref.size(); id++)
							{
								if (!cmpf(b[id].s[0], h_newVelocity_ref[id].s[0])||
									!cmpf(b[id].s[1], h_newVelocity_ref[id].s[1])||
									!cmpf(b[id].s[2], h_newVelocity_ref[id].s[2])||
									!cmpf(b[id].s[3], h_newVelocity_ref[id].s[3]))
								{
									std::cerr << "######### Something wrong at pos:\t" << id << std::endl;
									std::cerr << b[id].s[0] << " != " << h_newVelocity_ref[id].s[0] << std::endl;
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
