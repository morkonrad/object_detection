#pragma once
#include <iostream>
#include <array>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>



auto header = []()
{
	std::cout << "Benchmark,ElapsedTime[ms],Offload\n";
};

auto label_offload = [](const float off)->std::string
{
	if (off == 0.0)return"CPU_Only";
	if (int(off) >= 1) return "GPU_Only";

	std::stringstream ss;
	ss << std::to_string(int(off*100.0)) << "/" << std::to_string(100 - int(off * 100));
	return ss.str();

};

inline void on_error(const int err)
{
    if(err!=0)
    {
        std::cerr<<"Some error:"<<err<<std::endl;
        std::exit(err);
    }
}

inline void init_random(std::vector<std::uint8_t>& conatiner)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis_int(0, 255);
	for (auto& val : conatiner)
		val = dis_int(gen);
}

inline void init_random(std::vector<int>& conatiner,const int begin,const int end)
{
	if (end < begin)return;

	std::random_device rd;  
	std::mt19937 gen(rd()); 
	std::uniform_int_distribution<> dis_int(begin, end);
	for (auto& val : conatiner)
		val = dis_int(gen);
}

inline void init_random(std::vector<float>& conatiner)
{
	std::random_device rd;  
	std::mt19937 gen(rd()); 
	std::uniform_real_distribution<> dis_real(0.01, 1.0);
	for (auto& val : conatiner)
		val = static_cast<float>(dis_real(gen));
}

/**
 * @brief generate_offload_range
 * generates range with values from: <0.0f,step,...,1.0>
 * @param offloads
 * @param step
 */
inline void generate_offload_range(std::vector<float>& offloads, const float step)
{
	float v = -step;
    offloads.resize(size_t((1.0f / step) + 2));
	std::generate(offloads.begin(), offloads.end(), [&v, step]() {return v += step; });
	offloads.erase(std::remove_if(offloads.begin(), offloads.end(), [](float val) {
		return (val - 1.0f > 0.05f); }), offloads.end());
	std::replace_if(offloads.begin(), offloads.end(), [](float val) {return val > 1.0f; }, 1.0f);
}
