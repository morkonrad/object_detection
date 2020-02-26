#pragma once

//Minimum cl_hpp supported version
#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 200
#endif

// cl2.hpp is newest c++ bindings header from Khronos
// check: https://github.com/KhronosGroup/OpenCL-CLHPP to download it
// or sudo apt install opencl-headers
#include <CL/cl2.hpp>

#include <cmath>
#include <memory>
#include <map>
#include <mutex>
#include "clTaskFormat.h"

//#define _PROFILE_CLTASK_	
#ifdef _PROFILE_CLTASK_
#define _TASK_INSTRUMENTATION_
#include <chrono>
#include <utility>
#include <fstream>
#include <cmath>
#endif

inline bool cmpf(const float A,const float B,const float epsilon = 0.005f)
{
    return (std::fabs(A - B) < epsilon);
}

namespace clArgInfo
{

inline size_t cnt_items(const std::string& type)
{
    if (type.find('2') != std::string::npos)return 2;
    if (type.find('3') != std::string::npos)return 3;
    if (type.find('4') != std::string::npos)return 4;
    if (type.find('8') != std::string::npos)return 8;
    if (type.find("16") != std::string::npos)return 16;
    return 1;
}

template<typename T1, typename T2>
void cast_to_bytes(std::vector<int8_t>& bytes,
                   const size_t items,
                   const double init_value)
{
    const T1 v = { { static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value),
                     static_cast<T2>(init_value) } };

    bytes.resize(sizeof(T2)*items, 0);
    memcpy(bytes.data(), &v, bytes.size());
}

inline std::vector<int8_t>
get_byte_value(const std::string &type, const double init_value)
{
    std::vector<int8_t> bytes;
    if (type.empty())return bytes;

    const auto items = cnt_items(type);

    if (type.find("char") != std::string::npos)
    {
        cast_to_bytes<cl_char16, cl_char>(bytes, items, init_value);
    }
    else if (type.find("uchar") != std::string::npos)
    {
        cast_to_bytes<cl_uchar16, cl_uchar>(bytes, items, init_value);
    }
    else if (type.find("short") != std::string::npos)
    {
        cast_to_bytes<cl_short16, cl_short>(bytes, items, init_value);
    }
    else if (type.find("ushort") != std::string::npos)
    {
        cast_to_bytes<cl_ushort16, cl_ushort>(bytes, items, init_value);
    }
    else if (type.find("int") != std::string::npos)
    {
        cast_to_bytes<cl_int16, cl_int>(bytes, items, init_value);
    }
    else if (type.find("uint") != std::string::npos)
    {
        cast_to_bytes<cl_uint16, cl_uint>(bytes, items, init_value);
    }
    else if (type.find("long") != std::string::npos)
    {
        cast_to_bytes<cl_long16, cl_long>(bytes, items, init_value);
    }
    else if (type.find("ulong") != std::string::npos)
    {
        cast_to_bytes<cl_ulong16, cl_ulong>(bytes, items, init_value);
    }
    else if (type.find("half") != std::string::npos)
    {
        //cast_to_bytes<cl_half,cl_long>(bytes,items,init_value);
        cast_to_bytes<cl_uint16, cl_uint>(bytes, items, init_value);
    }
    else if (type.find("float") != std::string::npos)
    {
        cast_to_bytes<cl_float16, cl_float>(bytes, items, init_value);
    }
    else if (type.find("double") != std::string::npos)
    {
        cast_to_bytes<cl_double16, cl_double>(bytes, items, init_value);
    }

    return bytes;
}

inline size_t get_size(const std::string &type)
{
    if (type.empty())return 0;

    const auto items = cnt_items(type);

    if (type.find("char") != std::string::npos)
    {
        return sizeof(cl_char)*items;
    }
    if (type.find("uchar") != std::string::npos)
    {
        return sizeof(cl_uchar)*items;
    }
    if (type.find("short") != std::string::npos)
    {
        return sizeof(cl_short)*items;
    }
    else if (type.find("ushort") != std::string::npos)
    {
        return sizeof(cl_ushort)*items;
    }
    else if (type.find("int") != std::string::npos)
    {
        return sizeof(cl_int)*items;
    }
    else if (type.find("uint") != std::string::npos)
    {
        return sizeof(cl_uint)*items;
    }
    else if (type.find("long") != std::string::npos)
    {
        return sizeof(cl_long)*items;
    }
    else if (type.find("ulong") != std::string::npos)
    {
        return sizeof(cl_ulong)*items;
    }
    else if (type.find("half") != std::string::npos)
    {
        return sizeof(cl_half)*items;
    }
    else if (type.find("float") != std::string::npos)
    {
        return sizeof(cl_float)*items;
    }
    else if (type.find("double") != std::string::npos)
    {
        return sizeof(cl_double)*items;
    }

    return 1;
}

inline bool isFloat(const std::string& type)
{
    if (type.find("half") != std::string::npos)
        return true;
    if (type.find("float") != std::string::npos)
        return true;
    if (type.find("double") != std::string::npos)
        return true;

    return false;
}

}

namespace coopcl
{
static auto check_svm_support = [](int iflag, cl_device_id dev)->std::string
{
    if (dev == nullptr)return "";

    cl_device_svm_capabilities caps;

    cl_int err = clGetDeviceInfo(
                dev,
                CL_DEVICE_SVM_CAPABILITIES,
                sizeof(cl_device_svm_capabilities),
                &caps,
                nullptr
                );

    if (err != CL_SUCCESS)return"";

    std::string sflag;
    switch (iflag)
    {
    case CL_DEVICE_SVM_FINE_GRAIN_BUFFER:
        sflag = "CL_DEVICE_SVM_FINE_GRAIN_BUFFER";
        break;
    case CL_DEVICE_SVM_COARSE_GRAIN_BUFFER:
        sflag = "CL_DEVICE_SVM_COARSE_GRAIN_BUFFER";
        break;
    case CL_DEVICE_SVM_ATOMICS:
        sflag = "CL_DEVICE_SVM_ATOMICS";
        break;
    case CL_DEVICE_SVM_FINE_GRAIN_SYSTEM:
        sflag = "CL_DEVICE_SVM_FINE_GRAIN_SYSTEM";
        break;
    }

    if (sflag.empty())return sflag;
    return ((caps & iflag) == iflag) ? sflag : "";
    //return "";
};

static std::string err_msg(const int err)
{
    switch (err)
    {
    case CL_DEVICE_NOT_FOUND:
        return  "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
        return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return  "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
        return  "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
        return  "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return  "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
        return  "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
        return  "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return  "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
        return  "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
        return  "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        return  "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        return  "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_COMPILE_PROGRAM_FAILURE:
        return  "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE:
        return  "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE:
        return  "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED:
        return  "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        return  "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case CL_INVALID_VALUE:
        return  "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
        return  "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
        return  "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
        return  "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
        return  "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
        return  "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
        return  "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
        return  "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
        return  "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return  "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
        return  "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
        return  "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
        return  "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
        return  "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
        return  "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return  "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
        return  "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
        return  "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
        return  "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
        return  "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
        return  "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
        return  "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
        return  "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
        return  "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
        return  "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
        return  "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
        return  "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
        return  "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
        return  "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
        return  "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
        return  "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
        return  "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
        return  "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return  "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
        return  "CL_INVALID_PROPERTY";
    case CL_INVALID_IMAGE_DESCRIPTOR:
        return  "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS:
        return  "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS:
        return  "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
        return  "CL_INVALID_DEVICE_PARTITION_COUNT";

    }

    return "Unknown err_msg";
}

static void on_cl_error(const int err)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "Some error:\t" << err_msg(err) << std::endl;
        std::exit(err);
    }
}

class clMemory
{
private:

    void* _data{ nullptr };
    size_t _items{ 0 };
    size_t _size{ 0 };
    bool _read_only{ false };
    int _flag{ CL_MEM_READ_WRITE };

    const cl::Context* p_ctx_cpu{ nullptr };
    const cl::Context* p_ctx_gpu{ nullptr };

    std::unique_ptr<cl::Buffer> _buff_cpu{ nullptr };
    std::unique_ptr<cl::Buffer> _buff_gpu{ nullptr };

    void _clalloc(const cl::Context& ctx_cpu,
                  const cl::Context& ctx_gpu)
    {
        p_ctx_cpu = &ctx_cpu;
        p_ctx_gpu = &ctx_gpu;

        auto flag = CL_MEM_READ_WRITE;

        if (_read_only) flag = CL_MEM_READ_ONLY;

        int err = 0;
        _buff_cpu = std::make_unique<cl::Buffer>(ctx_cpu, flag | CL_MEM_USE_HOST_PTR, _size, (void*)_data, &err);
        on_cl_error(err);

        _buff_gpu = std::make_unique<cl::Buffer>(ctx_gpu, flag | CL_MEM_USE_HOST_PTR, _size, (void*)_data, &err);
        on_cl_error(err);
    }

    template<typename T>
    void* _appalloc(const size_t items, const cl::Context& ctx_gpu)
    {
        _items = items;
        _size = items * sizeof(T);
        _flag = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER;
        if (_read_only) _flag = CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER;

        _data = clSVMAlloc(ctx_gpu(), _flag, _size, 0);
        if (_data == nullptr)throw std::runtime_error(" memory==nullptr -->Check driver !!");
        return _data;
    }

public:

    clMemory(const clMemory&) = delete;
    clMemory& operator=(const clMemory&) = delete;
    clMemory& operator=(clMemory&&) = delete;
    clMemory(clMemory&&) = delete;
    clMemory() = default;

    //allocate only
    template<typename T>
    clMemory(const cl::Context& ctx_cpu,
             const cl::Context& ctx_gpu,
             const size_t items,
             const bool read_only = false)
    {

        if (items == 0)return;
        _read_only = read_only;
        _appalloc<T>(items, ctx_gpu);
        _clalloc(ctx_cpu, ctx_gpu);
    }

    //allocate and initialize with val
    template<typename T>
    clMemory(
            const cl::Context& ctx_cpu,
            const cl::Context& ctx_gpu,
            const size_t items,
            const T val, const bool read_only = false)
    {

        if (items == 0)return;
        _read_only = read_only;
        _appalloc<T>(items, ctx_gpu);

        for (size_t i = 0; i < items; i++)
            static_cast<T*>(_data)[i] = val;

        _clalloc(ctx_cpu, ctx_gpu);
    }

    //allocate and copy from src
    template<typename T>
    clMemory(
            const cl::Context& ctx_cpu,
            const cl::Context& ctx_gpu,
            const size_t items, const T* src, const bool read_only = false)
    {
        if (items == 0)return;
        if (src == nullptr)return;
        _read_only = read_only;
        _appalloc<T>(items, ctx_gpu);
        std::memcpy(_data, src, items * sizeof(T));
        _clalloc(ctx_cpu, ctx_gpu);
    }

    template<typename T>
    clMemory(
            const cl::Context& ctx_cpu,
            const cl::Context& ctx_gpu,
            const size_t items, const void* src, const bool read_only = false)
    {
        if (items == 0)return;
        if (src == nullptr)return;
        _read_only = read_only;
        _appalloc<T>(items, ctx_gpu);
        std::memcpy(_data, src, items * sizeof(T));
        _clalloc(ctx_cpu, ctx_gpu);
    }

    //allocate and copy from src
    template<typename T>
    clMemory(
            const cl::Context& ctx_cpu,
            const cl::Context& ctx_gpu,
            const std::vector<T>& src, const bool read_only = false)
    {
        if (src.empty())return;
        _read_only = read_only;
        const auto bytes = src.size() * sizeof(T);
        _appalloc<T>(src.size(), ctx_gpu);
        std::memcpy(_data, src.data(), bytes);
        _clalloc(ctx_cpu, ctx_gpu);
    }

    cl_mem get_mem()const
    {
        if (p_ctx_cpu)
            return (*_buff_cpu)();
        return (*_buff_gpu)();
    }

    cl_mem get_mem(const cl::Context& ctx)const
    {
        if (&ctx == p_ctx_cpu)
            return (*_buff_cpu)();

        return (*_buff_gpu)();
    }

    ~clMemory()
    {
        clSVMFree((*p_ctx_gpu)(), _data);

#ifdef _DEBUG
        cl_uint cnt_ref_cpu = 0;
        clGetMemObjectInfo((*_buff_cpu)(), CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &cnt_ref_cpu, nullptr);

        cl_uint cnt_ref_gpu = 0;
        clGetMemObjectInfo((*_buff_gpu)(), CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &cnt_ref_gpu, nullptr);
        if (cnt_ref_gpu != 1 || cnt_ref_cpu != 1)
        {
            std::cerr << "Found mem_leak, fixme!!" << std::endl;
            throw std::runtime_error("Memory leak found !");
        }
#endif // _DEBUG
    }

    template<typename T>
    const T at(const size_t id)const
    {
        return *(static_cast<const T*>(_data) + id);
    }

    void* data()const {
        return _data;
    }

    size_t items()const {
        return _items;
    }

    size_t item_size()const {
        return _size / _items;
    }

    size_t size()const {
        return _size;
    }

    bool isRead_only()const {
        return _read_only;
    }

};

class clTask
{
#ifdef _PROFILE_CLTASK_
    struct profiling_record
    {
    private:
        std::string _name{ "" };
        std::vector<std::string> _arg_values{};

        float _offload{ 0 };
        float _abs_gpu_time{ 0 };
        float _abs_cpu_time{ 0 };

        float _relative_task_begin{ 0 };
        float _relative_task_end{ 0 };
        float _relative_task_duartion{ 0 };

		std::array<size_t, 3> _work_group_size{ 0,0,0 };
    public:
        profiling_record(){}

        bool is_valid()const{
            if(_abs_gpu_time==0.0f && _abs_cpu_time==0.0f)return false;
            return true;
        }

        profiling_record(
                std::string  name,
                std::vector<std::string> arg_values,
                const float offload,
                const float abs_gpu_time,
                const float abs_cpu_time,
                const float relative_task_begin,
                const float relative_task_end,
                const float relative_task_duartion,
				const std::array<size_t,3> work_group_size):
            _name(name),
            _arg_values(arg_values),
            _offload(offload),
            _abs_gpu_time(abs_gpu_time),
            _abs_cpu_time(abs_cpu_time),
            _relative_task_begin(relative_task_begin),
            _relative_task_end(relative_task_end),
            _relative_task_duartion(relative_task_duartion),
			_work_group_size(work_group_size){}

        std::array<size_t,3> global_sizes()const{
            return {static_cast<size_t>(std::atoi(_arg_values[0].c_str())),
                        static_cast<size_t>(std::atoi(_arg_values[1].c_str())),
                        static_cast<size_t>(std::atoi(_arg_values[2].c_str()))};
        }

        std::array<size_t,3> work_group_sizes()const{
            return _work_group_size;
        }

        float offload()const{return _offload;}

        float execution_time()const{return std::max(_abs_cpu_time,_abs_gpu_time);}

        std::string name()const{return _name;}

        const std::vector<std::string>& args()const{return _arg_values;}

        void write_csv_line(std::ostream& out)const
        {
            out << _name << ",";
            out << _arg_values.size() << ",";
            
			for (const auto& av : _arg_values)
                out << av << ",";

            out << _offload << ","
                << _abs_gpu_time <<  ","
                << _abs_cpu_time << ","
                << _relative_task_begin<< ","
                << _relative_task_end <<","
                << _relative_task_duartion << ","
				<< _work_group_size[0] << ","
				<< _work_group_size[1] << ","
				<< _work_group_size[2] << "\n";
        }

        void read_csv_line(const std::string& line_csv)
        {
            std::istringstream iss(line_csv);
            std::string item; size_t id = 0;

            //positional parser
            while (getline(iss, item,','))
            {
                if (id==0)
                {
                    _name = item;
                }
                else if(id==1)
                {
                    const auto args_cnt = std::atoi(item.c_str());
                    _arg_values.resize(args_cnt);
                }
                else if (id>1 && id < _arg_values.size() + 2)
                {
                    _arg_values[id-2] = item;
                }
                else
                {
                    if (id == _arg_values.size() + 2) { _offload = std::atof(item.c_str()); }
                    else if (id == _arg_values.size() + 3) { _abs_gpu_time = std::atof(item.c_str()); }
                    else if (id == _arg_values.size() + 4) { _abs_cpu_time = std::atof(item.c_str()); }
                    else if (id == _arg_values.size() + 5) { _relative_task_begin = std::atof(item.c_str()); }
                    else if (id == _arg_values.size() + 6) { _relative_task_end = std::atof(item.c_str()); }
                    else if (id == _arg_values.size() + 7) { _relative_task_duartion = std::atof(item.c_str()); }
					else if (id == _arg_values.size() + 8) { _work_group_size[0] = static_cast<size_t>(std::atoi(item.c_str())); }
					else if (id == _arg_values.size() + 9) { _work_group_size[1] = static_cast<size_t>(std::atoi(item.c_str())); }
					else if (id == _arg_values.size() + 10) { _work_group_size[2] = static_cast<size_t>(std::atoi(item.c_str())); }
                }
                id++;
            }
        }
    };

    const std::chrono::time_point<std::chrono::system_clock>* _app_start_time_point{ nullptr };
    std::vector<profiling_record> _task_profile_log;
    long _task_end_time_cpu_ms{ 0 };
    long _task_end_time_gpu_ms{ 0 };
#endif
    cl::Context _ctx_cpu;
    cl::Context _ctx_gpu;

    cl::Kernel _kernel_cpu;
    cl::Kernel _kernel_gpu;

    std::mutex _user_event_mutex;
    std::mutex _observation_mutex;

    //event in cpu context
    cl::Event _cpu_ready;
    //_gpu_ready associated via callback with ctx_cpu
    cl::UserEvent _gpu_ready_cpu_ctx;

    //event in gpu context
    cl::Event _gpu_ready;
    //_cpu_ready associated via callback with ctx_gpu
    cl::UserEvent _cpu_ready_gpu_ctx;

    static constexpr auto _log_depth = 10;
    //size_t _counter_log{ 0 };

    //std::tuple 1: offload, 2: cpu_duration, 3: gpu_duration
    std::vector<std::tuple<float, float,float>> _previous_observation;
    std::vector<clTask*> _dependence_list;

    double _execution_time_cpu_msec{ 0 };
    double _execution_time_gpu_msec{ 0 };
    float _last_offload{ 0.0 };

    std::vector<std::string> _last_task_call_arg_values{};
	std::array<size_t,3> _last_task_call_group_size;

    struct arg_info
    {
        std::string _type_name{""}; // type_name(string,float ..)
        size_t _type_size{0}; //size in byte
        size_t _CL_KERNEL_ARG_ADDRESS_QUALIFIER{0}; //global,local,private
        size_t _CL_KERNEL_ARG_TYPE_QUALIFIER{0}; //const,volatile
        arg_info(){}
        arg_info(const std::string& type_name,
                 const size_t& type_size,
                 const size_t& arg_addr_qual,
                 const size_t& arg_type_qual) :_type_name(type_name),
            _type_size(type_size),
            _CL_KERNEL_ARG_ADDRESS_QUALIFIER(arg_addr_qual),
            _CL_KERNEL_ARG_TYPE_QUALIFIER(arg_type_qual){}

        bool is_valid()const{
            if(_CL_KERNEL_ARG_ADDRESS_QUALIFIER==0)return false;
            if(_type_name.empty())return false;
            return true;
        }

        void write_csv_line(std::ostream& out)const
        {
            out << _type_name << ","
                << _type_size << ","
                << _CL_KERNEL_ARG_TYPE_QUALIFIER << ","
                << _CL_KERNEL_ARG_ADDRESS_QUALIFIER << "\n";
        }

        void read_csv_line(const std::string& line_csv)
        {
            std::istringstream iss(line_csv);
            std::string item; size_t id = 0;
            while (getline(iss, item, ','))
            {
                switch (id)
                {
                case 0: {_type_name = item; }break;
                case 1: {_type_size = std::atoi(item.c_str()); }break;
                case 2: {_CL_KERNEL_ARG_TYPE_QUALIFIER = std::atoi(item.c_str()); }break;
                case 3: {_CL_KERNEL_ARG_ADDRESS_QUALIFIER = std::atoi(item.c_str()); }break;
                default:
                    break;
                }
                id++;
            }
        }
    };
    std::vector<arg_info> _arg_infos;

    std::string _body{ "" };
    std::string _name{ "" };
    std::string _jit_flags{ " " };
    std::uint8_t _dim_id_divided{ 0 };

    int build_args_info(const cl::Kernel* k)
    {
        if (k == nullptr)return -1;

        int err = 0;
        std::string err_log;
        size_t args = k->getInfo<CL_KERNEL_NUM_ARGS>(&err);
        if (err != CL_SUCCESS) {
            err_log = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            std::cerr << err_log << std::endl;
            return err;

        }

        for (cl_uint id = 0;id < args;id++)
        {
            const size_t aq = k->getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(id, &err);
            if (err != CL_SUCCESS) {
                err_log = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
                std::cerr << err_log << std::endl;
                return err;
            }

            cl_kernel_arg_type_qualifier tq;
            err = k->getArgInfo(id, CL_KERNEL_ARG_TYPE_QUALIFIER, &tq);
            if (err != CL_SUCCESS) {
                err_log = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
                std::cerr << err_log << std::endl;
                return err;

            }

            std::string type_name = k->getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(id, &err);
            if (err != CL_SUCCESS) {
                err_log = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
                std::cerr << err_log << std::endl;
                return err;
            }

            const auto pos = type_name.find('\000');
            if (pos != std::string::npos)
                type_name.replace(pos, 4, "");

            size_t type_size = clArgInfo::get_size(type_name);

            _arg_infos.push_back(arg_info{ type_name,type_size,aq,(size_t)tq });
        }

        _last_task_call_arg_values.resize(_arg_infos.size());

        return 0;
    }

#ifdef _TASK_INSTRUMENTATION_

    int add_task_argValue(const std::uint8_t& id, const std::unique_ptr<clMemory>& first)
    {
        if (_last_task_call_arg_values.empty()||id >= _last_task_call_arg_values.size())return -1;
        _last_task_call_arg_values[id] =""; //clBuffers marked as empty strings
        //_last_task_call_arg_values[id] = std::to_string(first->size());
        return 0;
    }

    int add_task_argValue(const std::uint8_t& id, const clMemory& first)
    {
        if (_last_task_call_arg_values.empty() || id >= _last_task_call_arg_values.size())return -1;
        _last_task_call_arg_values[id] =""; //clBuffers marked as empty strings
        //_last_task_call_arg_values[id] = std::to_string(first.size());
        return 0;
    }

    template <typename T>
    int add_task_argValue(const std::uint8_t &id, const T &arg)
    {
        if (_last_task_call_arg_values.empty() || id >= _last_task_call_arg_values.size())return -1;
        _last_task_call_arg_values[id] = std::to_string(arg);
        return 0;
    }

    int add_task_arg_value(const std::uint8_t &id)const{return 0;}

    template <typename T, typename... Args>
    int add_task_arg_value(std::uint8_t &id, const T &first, const Args&... rest)
    {
        int err = 0;
        err = add_task_argValue(id, first);
        if (err != 0) return err;
        id++;
        return add_task_arg_value(id, rest...);
    }

    const std::string marker_calls{ "_CALLS_" };
    const std::string marker_body{ "_BODY_" };
    const std::string marker_jit{ "_JIT_" };
    const std::string marker_args{ "_ARGS_" };

#endif

public:

    clTask(const clTask&) = delete;
    clTask& operator=(const clTask&) = delete;
    clTask& operator=(clTask&& ) = delete;
    clTask(clTask&&) = delete;
    clTask()= default;

#ifdef _PROFILE_CLTASK_		
    void set_app_begin_time_point(const std::chrono::time_point<std::chrono::system_clock>* app_begin)
    {
        _app_start_time_point = app_begin;
    }

    const std::chrono::time_point<std::chrono::system_clock>* get_app_begin_time_point()const  {
        return _app_start_time_point;
    }
#endif //_PROFILE_CLTASK_

#ifdef _TASK_INSTRUMENTATION_		

    void write_task_profile(std::ostream& out)const
    {
        wait();
        //----------------
        out << marker_calls<<"\n";
        //----------------
        for (auto& task_prof : _task_profile_log)
            task_prof.write_csv_line(out);

        //----------------
        out << marker_body<<"\n";
        //----------------
        int err = 0;
        auto prog = _kernel_cpu.getInfo<CL_KERNEL_PROGRAM>(&err);
        if (err != 0)return;
        auto body = prog.getInfo<CL_PROGRAM_SOURCE>(&err);
        if (err != 0)return;
        out << body << "\n";

        //----------------
        out << marker_jit<<"\n";
        //----------------
        out << _jit_flags << "\n";

        //----------------
        out << marker_args<<"\n";
        //----------------
        for (auto inf : _arg_infos)
            inf.write_csv_line(out);

    }

    int read_task_profile(const std::string& path_log_file)
    {
        if (path_log_file.empty())return -1;
        std::ifstream ifs(path_log_file);
        if (!ifs.is_open())return -1;

        const std::string file_txt(
                    (std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());

        //std::cout << file_txt << std::endl;
        const auto begin_calls = file_txt.find(marker_calls);
        const auto begin_body = file_txt.find(marker_body);
        const auto begin_jit = file_txt.find(marker_jit);
        const auto begin_args = file_txt.find(marker_args);

        const auto len_calls_section = begin_body - begin_calls;
        const auto len_body_section = begin_jit-begin_body;
        const auto len_jit_section = begin_args-begin_jit;
        const auto len_arg_section = file_txt.size()-begin_args;

        //cut sections
        std::string section_calls = file_txt.substr(begin_calls,len_calls_section);
        std::string section_body = file_txt.substr(begin_body, len_body_section);
        std::string section_jit = file_txt.substr(begin_jit, len_jit_section);;
        std::string section_args = file_txt.substr(begin_args, len_arg_section);

        auto remove_markers = [](std::string& section,const std::string& marker)->void
        {
            if (section.empty())return;
            const auto pos = section.find(marker);
            if (pos != std::string::npos)
                section.replace(pos, pos + marker.size(), "");
        };

        remove_markers(section_calls, marker_calls);
        remove_markers(section_body, marker_body);
        remove_markers(section_jit, marker_jit);
        remove_markers(section_args, marker_args);

        if(!section_calls.empty())
        {
            std::string line;
            std::istringstream iss(section_calls);
            while (std::getline(iss, line))
            {
                if (line.empty())continue;
                profiling_record pr;
                pr.read_csv_line(line);
                if(pr.is_valid())
                    _task_profile_log.push_back(pr);
            }
        }

        _body = section_body;
        _jit_flags = section_jit;

        if(!section_args.empty())
        {
            std::string line;
            std::istringstream iss(section_args);
            while (std::getline(iss, line))
            {
                if (line.empty())continue;
                arg_info ai;
                ai.read_csv_line(line);
                if(ai.is_valid())
                    _arg_infos.push_back(ai);
            }
        }

        if(!_task_profile_log.empty())
            _name =_task_profile_log.begin()->name();

        return 0;
    }

    const std::vector<profiling_record>& profiling_records()const{return _task_profile_log;}

    const std::vector<arg_info>& arg_infos()const{return _arg_infos;}

#endif // _TASK_INSTRUMENTATION_

    int build(
            const cl::Kernel& kernel_cpu,
            const cl::Kernel& kernel_gpu,
            const std::string& task_body,
            const std::string& task_name,
            const std::string& task_jit_flags = "")
    {
        _body = task_body;
        _name = task_name;
        _jit_flags = task_jit_flags;

        //Set device dependent context
        int err = 0;
        _kernel_cpu = kernel_cpu;
        _ctx_cpu = _kernel_cpu.getInfo<CL_KERNEL_CONTEXT>(&err);
        if (err != 0)return err;

        _kernel_gpu = kernel_gpu;
        _ctx_gpu = _kernel_gpu.getInfo<CL_KERNEL_CONTEXT>(&err);
        if (err != 0)return err;

        //Extract information about args
        return build_args_info(&_kernel_cpu);
    }

    int build(
            const cl::Kernel&  /*kernel_cpu*/,
            const cl::Kernel&  /*kernel_gpu*/,
            const std::string&  /*task_body_cpu*/,
            const std::string&  /*task_name_cpu*/,
            const std::string&  /*task_body_gpu*/,
            const std::string&  /*task_name_gpu*/,
            const std::string&  /*task_jit_flags_cpu*/ = "",
            const std::string&  /*task_jit_flags_gpu*/ = "")
    {
        // TODO:: Need to implement !!
        /*_body = task_body;
            _name = task_name;
            _jit_flags = task_jit_flags;

            //Set device dependent context
            int err = 0;
            _kernel_cpu = kernel_cpu;
            _ctx_cpu = _kernel_cpu.getInfo<CL_KERNEL_CONTEXT>(&err);
            if (err != 0)return err;

            _kernel_gpu = kernel_gpu;
            _ctx_gpu = _kernel_gpu.getInfo<CL_KERNEL_CONTEXT>(&err);
            if (err != 0)return err;

            //Extract information about args
            return build_args_info(&_kernel_cpu);
            */
        return 0;
    }

    ~clTask() {
        wait();
    }

    std::uint8_t get_dim_id_to_divide()const { return _dim_id_divided; };
    void set_ndr_dim_to_divide(const std::uint8_t dim_id = 0) { _dim_id_divided = dim_id; }

    std::string body()const { return _body; }
    std::string name()const { return _name; }
    std::string jit_flags()const { return _jit_flags; }

    const cl::Kernel* kernel_cpu()const { return &_kernel_cpu; }
    const cl::Kernel* kernel_gpu()const { return &_kernel_gpu; }

    cl::Event* cpu_ready(){ return &_cpu_ready; }
    cl::Event* gpu_ready(){ return &_gpu_ready; }

    const cl::Event* cpu_ready()const { return &_cpu_ready; }
    const cl::Event* gpu_ready()const { return &_gpu_ready; }

    const cl::UserEvent* cpu_user_ready_ctx_gpu()const { return &_cpu_ready_gpu_ctx; }
    const cl::UserEvent* gpu_user_ready_ctx_cpu()const { return &_gpu_ready_cpu_ctx; }

    float get_mean_exec_time()
    {
        float mean_exec_time_cpu_gpu = 0.0f;
        size_t obs_id = 1;
        for (auto& obs : _previous_observation)
        {
            const auto cpu_time = std::get<1>(obs);
            const auto gpu_time = std::get<2>(obs);
            mean_exec_time_cpu_gpu = cpu_time + gpu_time;
            obs_id++;
        }
        _previous_observation.clear();
        mean_exec_time_cpu_gpu = mean_exec_time_cpu_gpu / obs_id;
        return mean_exec_time_cpu_gpu;
    }

    int create_cpu_user_ready_ctx_gpu(const float offload)
    {
        int err = 0;
        if (_cpu_ready_gpu_ctx() != nullptr)
        {
            err = _cpu_ready_gpu_ctx.wait();
            if (err != 0)return err;
        }
        _cpu_ready_gpu_ctx = cl::UserEvent(_ctx_gpu, &err);
        if (err != 0)return err;
        _last_offload = offload;
        return err;
    }

    int create_gpu_user_ready_ctx_cpu(const float offload)
    {
        int err = 0;
        if (_gpu_ready_cpu_ctx() != nullptr)
        {
            err = _gpu_ready_cpu_ctx.wait();
            if (err != 0)return err;
        }
        _gpu_ready_cpu_ctx = cl::UserEvent(_ctx_cpu, &err);
        if (err != 0)return err;
        _last_offload = offload;
        return err;
    }

    float update_offload()const
    {
        if (_previous_observation.empty()) {
            return 0.5f;
        }

        const auto tuple = _previous_observation.back();
        const auto last_offload = std::get<0>(tuple);

        const auto cpu_duration = std::get<1>(tuple);
        if (cpu_duration == 0)return 1.0f;

        const auto gpu_duration = std::get<2>(tuple);
        if (gpu_duration == 0)return 0.0f;

        float updated_offload = 0;
        const auto ratio = cpu_duration / gpu_duration;
        if (std::fabs(1.0 - ratio) < 0.125)
            updated_offload = last_offload;
        else
            updated_offload = ratio < 1.0 ? last_offload - 0.0125f*ratio : ratio > 1.0 ? last_offload + 0.0125f*ratio : last_offload;

        if (updated_offload < 0) updated_offload = 0;
        if (updated_offload > 1) updated_offload = 1;

        return updated_offload;
    }

    std::string offload_str(const float off)
    {
        const int proc = static_cast<int>(off*100.0);
        std::stringstream str;
        str << "Offload to GPU " << std::to_string(proc) << "%";
        return str.str();
    }

    void set_async_task_duration(cl_event ev, const double duration)
    {
        std::lock_guard<std::mutex> guard(_observation_mutex);

        if (ev == _cpu_ready())
            _execution_time_cpu_msec = duration;
        else if (ev == _gpu_ready())
            _execution_time_gpu_msec = duration;

#ifdef _PROFILE_CLTASK_
        const auto task_end = std::chrono::system_clock::now();
        const auto elapsed_time_since_app_start_ms = _app_start_time_point == nullptr ? 0 : std::chrono::duration_cast<std::chrono::milliseconds>(task_end - *_app_start_time_point).count();

        if (ev == _cpu_ready())
            _task_end_time_cpu_ms = elapsed_time_since_app_start_ms;
        else if (ev == _gpu_ready())
            _task_end_time_gpu_ms = elapsed_time_since_app_start_ms;
#endif

        //std::tuple 1: offload, 2: cpu_duration, 3: gpu_duration
        if (_last_offload == 0.0 || _last_offload == 1.0)
        {
            _previous_observation.emplace_back( _last_offload,_execution_time_cpu_msec,_execution_time_gpu_msec );
#ifdef _PROFILE_CLTASK_
            //get time_point relative to the app_begin
            const auto task_duration = _last_offload == 0.0f ? _execution_time_cpu_msec : _execution_time_gpu_msec;
            const auto task_begin = elapsed_time_since_app_start_ms - task_duration;
            const auto task_end = elapsed_time_since_app_start_ms;

            profiling_record tlog(
                        _name,
                        _last_task_call_arg_values,
                        _last_offload,
                        (float)_execution_time_gpu_msec,
                        (float)_execution_time_cpu_msec,
                        task_begin,
                        task_end,
                        task_duration,_last_task_call_group_size);
            _task_profile_log.push_back(tlog);
#endif
            _execution_time_cpu_msec = 0;
            _execution_time_gpu_msec = 0;
        }
        else if (_execution_time_cpu_msec > 0 && _execution_time_gpu_msec > 0)
        {
            _previous_observation.emplace_back( _last_offload, _execution_time_cpu_msec, _execution_time_gpu_msec );
#ifdef _PROFILE_CLTASK_				
            // calc. time_point relative to the app_begin
            // Case parallel-overlap cpu/gpu processing:
            const auto begin_cpu_task = _task_end_time_cpu_ms - _execution_time_cpu_msec;
            const auto begin_gpu_task = _task_end_time_gpu_ms - _execution_time_gpu_msec;
            //Take the one that starts first
//#ifdef __linux
            const auto task_begin = std::min(begin_cpu_task, begin_gpu_task);
            const auto task_end = std::max(_task_end_time_cpu_ms, _task_end_time_gpu_ms);
//#else
//			const auto task_begin = std::min(begin_cpu_task, begin_gpu_task);
//            const auto task_end = std::max(_task_end_time_cpu_ms, _task_end_time_gpu_ms);
//#endif
            //Take the one that ends later
            const float task_duration = task_end - task_begin;
            profiling_record tlog(
                        _name,
                        _last_task_call_arg_values,
                        _last_offload,
                        (float)_execution_time_gpu_msec,
                        (float)_execution_time_cpu_msec,
                        task_begin,
                        task_end,
                        task_duration,_last_task_call_group_size);
            _task_profile_log.push_back(tlog);
#endif				
            _execution_time_cpu_msec = 0;
            _execution_time_gpu_msec = 0;
        }
    }

    int set_async_event_user_complete(cl_event ev)
    {
        std::lock_guard<std::mutex> guard(_user_event_mutex);

        if (ev == _cpu_ready())
            return _cpu_ready_gpu_ctx.setStatus(CL_COMPLETE);

        if (ev == _gpu_ready())
            return _gpu_ready_cpu_ctx.setStatus(CL_COMPLETE);

        return CL_INVALID_EVENT;
    }

    const std::vector<clTask*>& dependence_list()const{ return _dependence_list; }

    void add_dependence(clTask* task) { _dependence_list.push_back(task); }

    size_t get_arg_type_size(const size_t id)const
    {
        return clArgInfo::get_size(_arg_infos[id]._type_name);
    }

    size_t get_arg_count()const
    {
        return _arg_infos.size();
    }

    bool is_arg_buffer(const size_t id)const
    {
        switch (_arg_infos[id]._CL_KERNEL_ARG_ADDRESS_QUALIFIER)
        {
        case CL_KERNEL_ARG_ADDRESS_PRIVATE:
            return false;
        }
        return true;
    }

    bool is_arg_read_only(const size_t id)const
    {
        switch (_arg_infos[id]._CL_KERNEL_ARG_TYPE_QUALIFIER)
        {
        case CL_KERNEL_ARG_TYPE_CONST:
            return true;
        }
        return false;
    }

    bool is_arg_LocalMem(const size_t id)const
    {
        switch (_arg_infos[id]._CL_KERNEL_ARG_ADDRESS_QUALIFIER)
        {
        case CL_KERNEL_ARG_ADDRESS_LOCAL:
            return true;
        }
        return false;
    }

    bool is_arg_type_Float(const size_t id)const
    {
        return clArgInfo::isFloat(_arg_infos[id]._type_name);
    }

    std::vector<std::int8_t>
    get_init_arg_value(const size_t id, const float val)const
    {
        return clArgInfo::get_byte_value(_arg_infos[id]._type_name, val);
    }

    int wait()const
    {
        int err = 0;

        if (_cpu_ready_gpu_ctx() != nullptr)
        {
            err = _cpu_ready_gpu_ctx.wait();
            if (err != 0)return err;
        }

        if (_gpu_ready_cpu_ctx() != nullptr)
        {
            err = _gpu_ready_cpu_ctx.wait();
            if (err != 0)return err;
        }

        if (_gpu_ready() != nullptr)
        {
            err = _gpu_ready.wait();
            if (err != 0)return err;
        }

        if (_cpu_ready() != nullptr)
        {
            err = _cpu_ready.wait();
            if (err != 0)return err;
        }

        return err;
    }

    void write_records_to_stream(
            std::stringstream& ofs_offload,
            std::stringstream& ofs_cpu_time,
            std::stringstream& ofs_gpu_time)const
    {
        wait();
        size_t it = 1;
        ofs_offload << "iteration\toffload\n";
        ofs_cpu_time << "iteration\tcpu_time\n";
        ofs_gpu_time << "iteration\tgpu_time\n";
        for (const auto& observation : _previous_observation)
        {
            ofs_offload << it << "\t";
            ofs_offload << std::get<0>(observation) << "\n";

            ofs_cpu_time << it << "\t";
            ofs_cpu_time << std::get<1>(observation) << "\n";

            ofs_gpu_time << it++ << "\t";
            ofs_gpu_time << std::get<2>(observation) << "\n";
        }
    }

    template <typename... Args>
    int add_task_arg_values(
            const cl::NDRange& global_size,
            const cl::NDRange& work_group_size,
            const Args&... rest)
    {
        if (_last_task_call_arg_values.size() < 3|| global_size.dimensions()<3)return -1;

        //args 0,1,2 are reserved for Global_NDRange values
        std::uint8_t id = 0;
        for (;id<3;id++)
            _last_task_call_arg_values[id] = std::to_string(global_size[id]);

		_last_task_call_group_size = { 
			work_group_size[0],
			work_group_size[1],
			work_group_size[2] };

        return add_task_arg_value(id, rest ...);
    }
};

static void CL_CALLBACK user_ev_handler(cl_event ev, cl_int stat, void* user_data)
{
    int err = 0;
    auto ptr_Task = static_cast<clTask*>(user_data);
    if (ptr_Task == nullptr) {
        std::cerr << "Async_callback: couldn't read clTask, fixme!" << std::endl;
        return;
    }

    cl_ulong start = 0, end = 0;
    err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    on_cl_error(err);
    err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    on_cl_error(err);
    const auto duration = (cl_double)(end - start) * (cl_double)(1e-06);
    ptr_Task->set_async_task_duration(ev, duration);
    err = ptr_Task->set_async_event_user_complete(ev);
    if (err != 0) {
        on_cl_error(err);
        std::cerr << "Async_callback: couldn't set user_event status, fixme!" << std::endl;
    }
}

struct generic_arg
{
    std::unique_ptr<coopcl::clMemory> _clmem{ nullptr };
    std::vector<std::uint8_t> _arg_val;
    bool _isReadOnly;
    bool _isLocalMem;

    generic_arg(
            std::unique_ptr<coopcl::clMemory> clmem,
            std::vector<std::uint8_t> arg_val,
            bool isReadOnly=true,
            bool isLocalMem=false)
    {
        _clmem = std::move(clmem);
        _arg_val = arg_val;
        _isReadOnly = isReadOnly;
        _isLocalMem = isLocalMem;
    }

};

class ocl_device
{
private:
    cl::Context _ctx;
    cl::Device _device;
    std::vector<cl::CommandQueue> _queues;

    size_t _qid{ 0 };
    std::map<std::string, std::unique_ptr<cl::Program>> _bin_cache_programs;

    size_t				_max_work_group_size{ 0 };
    std::vector<size_t> _maximum_work_items_sizes;

    cl_device_type _dev_type;


    bool _hasUnified_mem{ false };
    bool _support_svm_fine_grain{ false };

    int SetArg(
            const cl::Context& ctx,
            const cl::Kernel &task,
            std::uint8_t &id,
            std::vector<generic_arg>& first)const
    {
        int err = 0;
        cl_kernel k = task();

        for (auto& arg : first)
        {
            if (arg._clmem != nullptr)
            {
                if (arg._isLocalMem)
                {
                    err = clSetKernelArg(k, id++, arg._clmem->size(), nullptr);
                    if (err != 0)return err;
                }
                else
                {
                    cl_mem  app_cl_mem = arg._clmem->get_mem(ctx);
                    err = clSetKernelArg(k, id++, sizeof(cl_mem), &app_cl_mem);
                    if (err != 0)return err;
                }
            }
            else
            {
                err = clSetKernelArg(k, id++, arg._arg_val.size(), arg._arg_val.data());
                if (err != 0)return err;
            }
        }
        return 0;
    }

    int SetArg(
            const cl::Context& ctx,
            const cl::Kernel &task,
            std::uint8_t &id,
            clMemory& first)const
    {
        cl_kernel k = task();
        cl_mem app_cl_mem = first.get_mem(ctx);
        return clSetKernelArg(k, id, sizeof(cl_mem), &app_cl_mem);
    }

    int SetArg(
            const cl::Context& ctx,
            const cl::Kernel &task,
            std::uint8_t &id,
            std::unique_ptr<clMemory>& first)const
    {
        cl_kernel k = task();
        cl_mem app_cl_mem = first->get_mem(ctx);
        return clSetKernelArg(k, id, sizeof(cl_mem), &app_cl_mem);
    }

    template <typename T>
    int SetArg(
            const cl::Context& ctx,
            const cl::Kernel &task,
            std::uint8_t &id,
            T &arg)const
    {
        cl_kernel k = task();
        return clSetKernelArg(k, id, sizeof(T), &arg);
    }

    int SetArgs(
            const cl::Context& ctx,
            const cl::Kernel &task,
            std::uint8_t &id)const
    {
        return 0;
    }

    template <typename T, typename... Args>
    int SetArgs(
            const cl::Context& ctx,
            const cl::Kernel &task,
            std::uint8_t &id,
            T &first,
            Args &... rest)const
    {
        int err = 0;
        err = SetArg(ctx, task, id, first);
        if (err != 0)
            return err;
        id++;
        return SetArgs(ctx, task, id, rest...);
    }

    cl::Kernel build(const std::string& task_name,
                     const std::string& task_name_cache,
                     int& err)
    {
        err = 0;
        if (_bin_cache_programs.empty())
        {
            err = -1;
            return cl::Kernel();
        }
        return cl::Kernel(clCreateKernel((*_bin_cache_programs.at(task_name_cache))(), task_name.c_str(), &err));
    }

    std::string calc_cache_name(const std::string& task_name, const std::array<size_t, 3>& global_size)const
    {
        std::stringstream task_cache_name;

        task_cache_name << task_name << "_"
                        << global_size[0] << "_"
                        << global_size[1] << "_"
                        << global_size[2];

        return task_cache_name.str();
    }

    int build_program(
            const std::string& rewriten_task,
            const std::string& task_cache_name,
            const std::string& options)
    {
        int err = 0;

        _bin_cache_programs[task_cache_name] = std::make_unique<cl::Program>(_ctx, rewriten_task, false, &err);
        on_cl_error(err);

        std::vector<cl::Device> devs{ _device };
        err = _bin_cache_programs[task_cache_name]->build(devs, options.c_str());
        if (err != CL_SUCCESS)
        {
            std::cerr << _bin_cache_programs[task_cache_name]->getBuildInfo<CL_PROGRAM_BUILD_LOG>(devs[0], &err) << std::endl;
            on_cl_error(err);
            return CL_BUILD_PROGRAM_FAILURE;
        }
        return err;
    }

    cl::Kernel add_build_task(
            const std::string& body,
            const std::string& name,
            const std::string& jit_flags)
    {
        const auto rewriten_task = rewrite::add_execution_guard_to_kernels(body);
        int err = build_program(rewriten_task, name, jit_flags);
        on_cl_error(err);

        auto task = cl::Kernel(clCreateKernel((*_bin_cache_programs.at(name))(), name.c_str(), &err));
        on_cl_error(err);

        return task;
    }
public:

    ocl_device(const ocl_device&) = delete;
    ocl_device& operator=(const ocl_device&) = delete;
    ocl_device& operator=(ocl_device&& ) = delete;
    ocl_device(ocl_device&&) = delete;
    ~ocl_device() = default;
    ocl_device(/*const cl::Platform& platform,*/ cl::Device& device)
    {
        int err = 0;
        _device = device;

        _dev_type = _device.getInfo<CL_DEVICE_TYPE>(&err);
        on_cl_error(err);

        _maximum_work_items_sizes = _device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>(&err);
        on_cl_error(err);

        _max_work_group_size = _device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);
        on_cl_error(err);

        //_hasUnified_mem = _device.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>(&err);
        on_cl_error(err);

        if (_dev_type == CL_DEVICE_TYPE_GPU)
        {
            const auto ret = check_svm_support(CL_DEVICE_SVM_FINE_GRAIN_BUFFER, _device());
            if (!ret.empty()) _support_svm_fine_grain = true;
        }

        _ctx = cl::Context(_device, nullptr, nullptr, nullptr, &err);
        on_cl_error(err);

        const auto cnt_cu = _device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&err);
        for (size_t i = 0; i < cnt_cu; i++) {
            _queues.emplace_back(_ctx, _device, CL_QUEUE_PROFILING_ENABLE, &err);
            //_queues.push_back(cl::CommandQueue(_ctx, _device, 0, &err));
            on_cl_error(err);
        }

    }

    template <typename... Args>
    int execute(
            const float offload,
            clTask& task,
            const cl::NDRange global_no_split,
            const cl::NDRange global,
            const cl::NDRange local,
            const cl::NDRange offset,
            Args&... rest)
    {
        int err = 0;
        std::uint8_t id = 0;

        auto kernel = _dev_type == CL_DEVICE_TYPE_CPU ? task.kernel_cpu() : task.kernel_gpu();
        cl::Event* event_wait = _dev_type == CL_DEVICE_TYPE_CPU ? task.cpu_ready() : task.gpu_ready();

        const int gx = global_no_split[0];
        const int gy = global_no_split[1];
        const int gz = global_no_split[2];
        // add_instrumentation
#ifdef _TASK_INSTRUMENTATION_
        err = task.add_task_arg_values(global_no_split,local, rest ...);
        on_cl_error(err);
#endif // _TASK_INSTRUMENTATION_
        err = SetArgs(_ctx, *kernel, id, gx, gy, gz, rest ...);
        on_cl_error(err);

        std::vector<cl::Event> wait_list;
        if (!task.dependence_list().empty())
        {
            for (auto t : task.dependence_list())
            {
                // wait for bot CPU and GPU, if someone is not busy than the wait_ops. is approx. nop
                // dependent on the _dev_type get an event from  CPU and GPU that are from different context
                // an event_user_wait is an event from different context, it means if the _dev_type is a CPU than get an event_GPU from other ctx.

                //if a dev_type is CPU than the event_user_wait is from GPU, else opposite
                const cl::UserEvent* event_user_wait = _dev_type == CL_DEVICE_TYPE_CPU ? t->gpu_user_ready_ctx_cpu() : t->cpu_user_ready_ctx_gpu();
                //if a dev_type is CPU than the event_wait is from CPU, for GPU same
                const cl::Event* event_wait = _dev_type == CL_DEVICE_TYPE_CPU ? t->cpu_ready() : t->gpu_ready();

                //Than always wait for CPU+GPU if any event exist!
                if((*event_user_wait)()!=nullptr)
                    wait_list.push_back(*event_user_wait);

                if ((*event_wait)() != nullptr)
                    wait_list.push_back(*event_wait);

                //t->wait();
            }
        }

        if (_qid >= _queues.size())_qid = 0;
        //#ifdef _DEBUG
        //			std::cout << "{gx,gy,gz}{" << global[0] << "," << global[1] << "," << global[2] << "}\n";
        //			std::cout << "{lx,ly,lz}{" << local[0] << "," << local[1] << "," << local[2] << "}\n";
        //#endif
        err = _queues[_qid].enqueueNDRangeKernel(*kernel, offset, global, local, &wait_list, event_wait);
        on_cl_error(err);

        if (_dev_type == CL_DEVICE_TYPE_CPU)
        {
            err = task.create_cpu_user_ready_ctx_gpu(offload);
            if(err!=0)return err;
        }
        else
        {
            err = task.create_gpu_user_ready_ctx_cpu(offload);
            if (err != 0)return err;
        }

        //Set host_async_event_callback
        event_wait->setCallback(CL_COMPLETE, &user_ev_handler, &task);
        return _queues[_qid++].flush();

    }

    int wait()const
    {
        int err = 0;
        for (auto& q : _queues) {
            err = q.finish();
            if (err != 0)return err;
        }
        return err;
    }

    bool has_svm()const { return _support_svm_fine_grain; }

    size_t maximum_work_group_size()const { return _max_work_group_size; }

    std::vector<size_t> maximum_work_items_sizes()const { return _maximum_work_items_sizes; }

    const cl::Context* ctx()const { return &_ctx; }

    cl::Kernel build_tasks(
            const std::string& body,
            const std::string& name,
            const std::string& jit_flags = "")
    {
        if (body.empty() || name.empty())
        {
            throw std::runtime_error("Task body or name need to be non empty, fixme!!!");
        }

        int err = 0;
        //Check map if task exist ? yes, return kernel
        for (auto& item : _bin_cache_programs)
        {
            if (item.first == name)
            {
                auto task = cl::Kernel(clCreateKernel((*item.second)(), name.c_str(), &err));
                on_cl_error(err);
                return task;
            }
        }

        // task not found then add into map
        return add_build_task(body, name, jit_flags);
    }

};

static auto round_to_wg_multiple = [](
        const cl::NDRange& global, const cl::NDRange& local_in,
        cl::NDRange& global_cpu, cl::NDRange& global_gpu,
        cl::NDRange& local_cpu, cl::NDRange& local_gpu,
        const size_t items_cpu, const size_t items_gpu,
        cl::NDRange& global_offset,
        const size_t dim_ndr, const size_t dim_to_split)->int
{
    //if any local_in[]==0 --> a case where RT can set any local_size
    //8 for dim_0 because of CPU vectorization
    cl::NDRange local = {
        local_in[0] == 0 ? 8 : local_in[0],
        local_in[1] == 0 ? 1 : local_in[1],
        local_in[2] == 0 ? 1 : local_in[2]};

    const auto loc_size = local[dim_to_split];
    const size_t wg_mul_cpu = items_cpu % loc_size;
    const size_t wg_mul_gpu = items_gpu % loc_size;

    const size_t gx_pad_cpu = (wg_mul_cpu == 0 ? items_cpu / loc_size : (items_cpu / loc_size) + 1)*loc_size;
    const size_t gx_pad_gpu = (wg_mul_gpu == 0 ? items_gpu / loc_size : (items_gpu / loc_size) + 1)*loc_size;

    switch (dim_ndr)
    {
    case 1:
        local_cpu = { local[0] };
        local_gpu = { local[0] };

        global_cpu = { gx_pad_cpu };
        global_gpu = { gx_pad_gpu };
        global_offset = { gx_pad_cpu };

        break;
    case 2:
        local_cpu = { local[0],local[1] };
        local_gpu = { local[0],local[1] };

        switch (dim_to_split)
        {
        case 0:
            global_cpu = { gx_pad_cpu,global[1] };
            global_gpu = { gx_pad_gpu,global[1] };
            global_offset = { gx_pad_cpu,0 };
            break;
        case 1:
            global_cpu = { global[0],gx_pad_cpu };
            global_gpu = { global[0],gx_pad_gpu };
            global_offset = { 0, gx_pad_cpu };
            break;
        case 2:
            global_cpu = { global[0],global[1],gx_pad_cpu };
            global_gpu = { global[0],global[1],gx_pad_gpu };
            global_offset = { 0, 0, gx_pad_cpu };
            break;
        }
        break;
    case 3:
        local_cpu = { local[0],local[1],local[2] };
        local_gpu = { local[0],local[1],local[2] };

        switch (dim_to_split)
        {
        case 0:
            global_cpu = { gx_pad_cpu,global[1],global[2] };
            global_gpu = { gx_pad_gpu,global[1],global[2] };
            global_offset = { gx_pad_cpu,0,0 };
            break;
        case 1:
            global_cpu = { global[0],gx_pad_cpu,global[2] };
            global_gpu = { global[0],gx_pad_gpu,global[2] };
            global_offset = { 0, gx_pad_cpu,0 };
            break;
        case 2:
            global_cpu = { global[0],global[1],gx_pad_cpu };
            global_gpu = { global[0],global[1],gx_pad_gpu };
            global_offset = { 0, 0, gx_pad_cpu };
            break;
        }
        break;
    }

    //calculate all WI requested and divided for CPU & GPU
    const auto wi_cpu_splited = global_cpu[0] + global_cpu[1] + global_cpu[2];
    const auto wi_gpu_splited = global_gpu[0] + global_gpu[1] + global_gpu[2];
    const auto wi_input = global[0] + global[1] + global[2];

    //If wi_cpu_splited after round to wg_multiple is more than all_wi then return 1 (means ==> no CPU/GPU "divided" execution)
    //If wi_gpu_splited after round to wg_multiple is more than all_wi then return 1 (means ==> no CPU/GPU "divided" execution)
    if (wi_cpu_splited >= wi_input)return 1;
    if (wi_gpu_splited >= wi_input)return 1;

    return 0;
};

class virtual_device
{

private:
    //different cpu, gpu context
    const cl::Context* _ctx_cpu{ nullptr };
    const cl::Context* _ctx_gpu{ nullptr };

    std::unique_ptr<ocl_device> _dGPU{ nullptr };
    std::unique_ptr<ocl_device> _dCPU{ nullptr };

    std::string _platform_name;

    static int divide_ndranges(
            const float offload,
            const cl::NDRange& global,
            const cl::NDRange& local,
            cl::NDRange& global_cpu,
            cl::NDRange& global_gpu,
            cl::NDRange& global_offset,
            cl::NDRange& local_cpu,
            cl::NDRange& local_gpu,
            const std::uint8_t dim_to_split=0)
    {
        const auto dim_ndr = static_cast<std::uint8_t>(global.dimensions());

        // dim_to_split is 0,1,2 and dim_ndr is 1,2,3
        // assert --> dim_to_split < dim_ndr !
        if (dim_to_split >= dim_ndr) return -1;

        size_t items = 1;
        for (size_t dim = 0; dim < dim_ndr; dim++)
            items *= global[dim];

        if (items < 1) return -1;

        const size_t dim_split = global[dim_to_split];

        const float one_item = (float)dim_split / 100.0f;
        const auto procent = offload * 100.0f; //offload range--> (0:1>

        auto items_gpu = static_cast<size_t>((procent * one_item) < dim_split ? std::ceil(procent * one_item) : dim_split);
        size_t items_cpu = dim_split - items_gpu;

        if (items_cpu < 4) { return 1; }//no offload/data-split, workload too small
        if (items_gpu < 4) { return 1; }//no offload/data-split, workload too small

        //------------------------------------------
        //group_sizes + global_sizes extend/pad
        //------------------------------------------
        return round_to_wg_multiple(global, local,
                                    global_cpu, global_gpu,
                                    local_cpu, local_gpu,
                                    items_cpu, items_gpu,
                                    global_offset, dim_ndr, dim_to_split);
    }
    //TODO: Impl. separate body,name,flags for CPU and GPU
    //		int build_tasks(
    //			clTask& task,
    //			const std::string& body_cpu,
    //			const std::string& name_cpu,
    //            const std::string& jit_flags_cpu,
    //            const std::string& body_gpu,
    //            const std::string& name_gpu,
    //            const std::string& jit_flags_gpu)
    //		{
    //			return -1;
    //		}

    int build_tasks(
            clTask& task,
            const std::string& body,
            const std::string& name,
            const std::string& jit_flags = "")
    {
        const auto kcpu = _dCPU->build_tasks(body, name, jit_flags);
        const auto kgpu = _dGPU->build_tasks(body, name, jit_flags);
        return task.build(kcpu, kgpu, body, name, jit_flags);
    }

public:
    ~virtual_device()=default;
    virtual_device(const virtual_device&) = delete;
    virtual_device& operator=(const virtual_device&) = delete;
    virtual_device& operator=(virtual_device&& ) = delete;
    virtual_device(virtual_device&&) = delete;
    virtual_device()
    {
        int err = 0;
        std::vector<cl::Platform> platforms;
        err = cl::Platform::get(&platforms);
        on_cl_error(err);

        //check if GPU or CPU is available
        std::vector<cl::Device> devs;
        for (const auto& p : platforms)
        {
            const std::string pname = p.getInfo<CL_PLATFORM_NAME>(&err);
            //std::cout << pname<< std::endl;

            p.getDevices(CL_DEVICE_TYPE_CPU|CL_DEVICE_TYPE_GPU, &devs);
            if (!devs.empty())
            {
                for (auto& d : devs)
                {
                    auto dt = d.getInfo<CL_DEVICE_TYPE>(&err);
                    on_cl_error(err);
                    if (dt == CL_DEVICE_TYPE_CPU && _dCPU == nullptr)
                    {
                        _dCPU = std::make_unique<ocl_device>(/*p,*/ d);
                        if (!_platform_name.empty())_platform_name.append("+");
                        _platform_name.append(pname);
                    }
                    else if (dt == CL_DEVICE_TYPE_GPU && _dGPU == nullptr)
                    {
                        _dGPU = std::make_unique<ocl_device>(/*p,*/ d);
                        if (!_platform_name.empty())_platform_name.append("+");
                        _platform_name.append(pname);
                    }


                }
            }
        }

        if (_dCPU == nullptr) throw std::runtime_error("Minimal requirement: CPU with OpenCL installed!  exit ...");
        if (_dGPU == nullptr) throw std::runtime_error("Minimal requirement: CPU+GPU with OpenCL installed!  exit ...");

        _ctx_gpu = _dGPU->ctx();
        //_ctx_gpu = _dCPU->ctx();
        _ctx_cpu = _dCPU->ctx();

        if(!_dGPU->has_svm())throw std::runtime_error("Minimal requirement: GPU with support for OpenCL2.x and SVM_FINE_GRAIN_BUFFER installed!  exit ...");

    }

    virtual_device(	const std::string& cpu_platform_name,
                    const std::string& gpu_platform_name)
    {
        int err = 0;
        std::vector<cl::Platform> platforms;
        err = cl::Platform::get(&platforms);
        on_cl_error(err);

        //check if GPU or CPU is available
        std::vector<cl::Device> devs;
        for (const auto& p : platforms)
        {
            const std::string pname = p.getInfo<CL_PLATFORM_NAME>(&err);
            //std::cout << pname << std::endl;
            p.getDevices(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, &devs);
            if (!devs.empty())
            {
                for (auto& d : devs)
                {
                    auto dt = d.getInfo<CL_DEVICE_TYPE>(&err);
                    on_cl_error(err);
                    if (dt == CL_DEVICE_TYPE_CPU && _dCPU == nullptr) {
                        if (pname.find(cpu_platform_name) != std::string::npos)
                        {
                            _dCPU = std::make_unique<ocl_device>(/*p,*/ d);

                            if (!_platform_name.empty())_platform_name.append("+");
                            _platform_name.append(pname);

                        }
                    }
                    else if (dt == CL_DEVICE_TYPE_GPU && _dGPU == nullptr) {
                        if (pname.find(gpu_platform_name) != std::string::npos)
                        {
                            _dGPU = std::make_unique<ocl_device>(/*p,*/ d);

                            if (!_platform_name.empty())_platform_name.append("+");
                            _platform_name.append(pname);
                        }
                    }
                }
            }
        }

        if (_dCPU == nullptr) throw std::runtime_error("Minimal requirement: CPU with OpenCL installed!  exit ...");
        if (_dGPU == nullptr) throw std::runtime_error("Minimal requirement: CPU+GPU with OpenCL installed!  exit ...");

        _ctx_gpu = _dGPU->ctx();
        _ctx_cpu = _dCPU->ctx();

        if (!_dGPU->has_svm())throw std::runtime_error("Minimal requirement: GPU with support for OpenCL2.x and SVM_FINE_GRAIN_BUFFER installed!  exit ...");

    }

    int build_task(
            clTask& task,
//            const std::array<size_t, 3>& global_size,
            const std::string& body,
            const std::string& name,
            const std::string& jit_flags = "")
    {
        if (body.empty() || name.empty()) return -100;
        return build_tasks(task, body, name, jit_flags);
    }

    int wait()const
    {
        int err = _dGPU->wait();
        if(err!=0)return err;

        return _dCPU->wait();
    }

    template <typename... Args>
    int execute_async(
            clTask& task,
            const float offload_,
            const std::array<size_t, 3> global,
            const std::array<size_t, 3> local,
            Args&... rest)
    {
        int err = 0;

        float offload = offload_;

        if(offload_==-1) offload = task.update_offload();

        if ((int)offload < 0)return-1;
        if ((int)offload > 1)return-1;

        cl::NDRange global_in{
            global[0] == 0 ? 1 : global[0],
                    global[1] == 0 ? 1 : global[1],
                    global[2] == 0 ? 1 : global[2]
        };

        cl::NDRange local_in = {
            local[0],
            local[1],
            local[2] };

        if (local[0] == 0 && local[1] == 0 && local[2] == 0)
            local_in = cl::NullRange;

        if (cmpf(offload, 1.0f))
        {
            return _dGPU->execute(offload, task, global_in, global_in, local_in, cl::NullRange, rest ...);
        }
        if (cmpf(offload, 0.0f))
        {
            return _dCPU->execute(offload,task, global_in, global_in, local_in, cl::NullRange, rest ...);
        }
        else
        {
            cl::NDRange gcpu, ggpu, offset, loc_cpu, loc_gpu;
            const auto dim_to_split_id = task.get_dim_id_to_divide();
            const auto res = divide_ndranges(offload, global_in, local_in, gcpu, ggpu, offset, loc_cpu, loc_gpu,dim_to_split_id);
            if (res == -1) { return CL_INVALID_OPERATION; }

            if (res == 1) //workload remainder is to small to execute on both devices
            {
                if (offload > 0.5f){ return _dGPU->execute(1.0,task, global_in, global_in, local_in, cl::NullRange, rest ...); }
                else{ return _dCPU->execute(0.0,task, global_in, global_in, local_in, cl::NullRange, rest ...); }
            }
            else
            {
                err = _dGPU->execute(offload, task, global_in, ggpu, loc_gpu, offset, rest ...);
                on_cl_error(err);

                err = _dCPU->execute(offload, task, global_in, gcpu, loc_cpu, cl::NullRange, rest ...);
                on_cl_error(err);
            }
        }

        return err;
    }

    template <typename... Args>
    int execute(
            clTask& task,
            const float offload,
            const std::array<size_t, 3> global,
            const std::array<size_t, 3> local,
            Args&... rest)
    {
        auto err = execute_async(task, offload, global, local, rest ...);
        on_cl_error(err);
        err = task.wait();
        return err;
    }

    std::unique_ptr<clMemory>
    alloc(const size_t items, const bool read_only = false)
    {
        return std::make_unique<clMemory>(*_ctx_cpu, *_ctx_gpu, items, 0, read_only);
    }

    //allocate and initialize with 0
    template<typename T>
    std::unique_ptr<clMemory>
    alloc(const size_t items, const bool read_only = false)
    {
        T dummy_zero{};
        std::memset(&dummy_zero, 0, sizeof(T));
        return std::make_unique<clMemory>(*_ctx_cpu,*_ctx_gpu, items, dummy_zero, read_only);
    }

    //allocate and initialize with val
    template<typename T>
    std::unique_ptr<clMemory>
    alloc(const size_t items, const T val, const bool read_only = false)
    {
        return std::make_unique<clMemory>(*_ctx_cpu, *_ctx_gpu, items, val, read_only);
    }

    //allocate and copy from src
    template<typename T>
    std::unique_ptr<clMemory>
    alloc( const size_t items, const T* src, const bool read_only = false)
    {
        return std::make_unique<clMemory>(*_ctx_cpu, *_ctx_gpu, items, src, read_only);
    }

    //allocate and copy from src
    template<typename T>
    std::unique_ptr<clMemory>
    alloc(const std::vector<T>& src, const bool read_only = false)
    {
        return std::make_unique<clMemory>(*_ctx_cpu, *_ctx_gpu, src, read_only);
    }

    size_t maximum_work_group_size()const
    {
        const auto mwgs1 = _dCPU->maximum_work_group_size();
        if (_dGPU == nullptr)return mwgs1;

        const auto mwgs2 = _dGPU->maximum_work_group_size();
        //return smaller one
        return mwgs1>mwgs2?mwgs2:mwgs1;
    }

    std::vector<size_t>
    maximum_work_items_sizes()const
    {
        const auto wi1 = _dCPU->maximum_work_items_sizes();
        if (_dGPU == nullptr)return wi1;
        const auto wi2 = _dGPU->maximum_work_items_sizes();

        std::vector<size_t> max_wi;
        //return smaller one
        for (size_t i = 0;i < wi1.size();i++)
        {
            if (wi1[i] >= wi2[i])
                max_wi.push_back(wi2[i]);
            else
                max_wi.push_back(wi1[i]);
        }
        return max_wi;
    }

    std::string platform_name()const
    {
        return _platform_name;
    }
};

}

