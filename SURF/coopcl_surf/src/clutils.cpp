 /****************************************************************************\
 * Copyright (c) 2011, Advanced Micro Devices, Inc.                           *
 * All rights reserved.                                                       *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 *                                                                            *
 * Redistributions of source code must retain the above copyright notice,     *
 * this list of conditions and the following disclaimer.                      *
 *                                                                            *
 * Redistributions in binary form must reproduce the above copyright notice,  *
 * this list of conditions and the following disclaimer in the documentation  *
 * and/or other materials provided with the distribution.                     *
 *                                                                            *
 * Neither the name of the copyright holder nor the names of its contributors *
 * may be used to endorse or promote products derived from this software      *
 * without specific prior written permission.                                 *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS        *
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED  *
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR *
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR          *
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,      *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,        *
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR         *
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF     *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING       *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         *
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.               *
 *                                                                            *
 * If you use the software (in whole or in part), you shall adhere to all     *
 * applicable U.S., European, and other export laws, including but not        *
 * limited to the U.S. Export Administration Regulations (“EAR”), (15 C.F.R.  *
 * Sections 730 through 774), and E.U. Council Regulation (EC) No 1334/2000   *
 * of 22 June 2000.  Further, pursuant to Section 740.6 of the EAR, you       *
 * hereby certify that, except pursuant to a license granted by the United    *
 * States Department of Commerce Bureau of Industry and Security or as        *
 * otherwise permitted pursuant to a License Exception under the U.S. Export  *
 * Administration Regulations ("EAR"), you will not (1) export, re-export or  *
 * release to a national of a country in Country Groups D:1, E:1 or E:2 any   *
 * restricted technology, software, or source code you receive hereunder,     *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such *
 * technology or software, if such foreign produced direct product is subject *
 * to national security controls as identified on the Commerce Control List   *
 *(currently found in Supplement 1 to Part 774 of EAR).  For the most current *
 * Country Group listings, or for additional information about the EAR or     *
 * your obligations under those regulations, please refer to the U.S. Bureau  *
 * of Industry and Security’s website at http://www.bis.doc.gov/.             *
 \****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include "eventlist.h"
#include "clutils.h"
#include "utils.h"

#ifdef  _ORG_CL_DRIVER
#include <CL/cl.h>
#else
#include "clDriver.h"
#include <fstream>
#include <map>
#include "surf_constants.h"
#endif 

#ifdef  _ORG_CL_DRIVER
// The following variables have file scope to simplify
// the utility functions

//! All discoverable OpenCL platforms
static cl_platform_id* platforms = NULL;
static cl_uint numPlatforms;

//! All discoverable OpenCL devices (one pointer per platform)
static cl_device_id** devices = NULL;
static cl_uint* numDevices;

//! The chosen OpenCL platform
static cl_platform_id platform = NULL;

//! The chosen OpenCL device
static cl_device_id device = NULL;

//! OpenCL context
static cl_context context = NULL;

//! OpenCL command queue
static cl_command_queue commandQueue = NULL;
static cl_command_queue commandQueueProf = NULL;
static cl_command_queue commandQueueNoProf = NULL;

//! List of precompiled kernels
static cl_kernel kernel_list[NUM_KERNELS];

//! List of program objects
static cl_program program_list[NUM_PROGRAMS];

//! Globally visible event table
static EventList* events = NULL;

//! Global status of events
static bool eventsEnabled = false;
#else
static std::unique_ptr<coopcl::virtual_device> _virtual_device = nullptr;

static coopcl::clTask task_descriptor_surf;
static coopcl::clTask task_orient1_surf;
static coopcl::clTask task_orient2_surf;
static coopcl::clTask task_hessianDet_surf;
static coopcl::clTask task_scan_surf;
static coopcl::clTask task_scan4_surf;
static coopcl::clTask task_scanImage_surf;
static coopcl::clTask task_transpose_surf;
static coopcl::clTask task_transposeImage_surf;
static coopcl::clTask task_NearestNeighbor_surf;
static coopcl::clTask task_non_max_supression_surf;
static coopcl::clTask task_normalizeDescriptors_surf;

static cl_kernel kernel_list[NUM_KERNELS];
static std::map<cl_mem, std::unique_ptr<coopcl::clMemory>> _memory_objects;
static std::map<cl_kernel, coopcl::clTask*> _tasks_surf;
static float _offload{ 0 };

void dump_tasks()
{
    return;
}

static void build_kernel_task_map(coopcl::clTask& task,
    const int kernel_id,const std::array<size_t,3>& global_size,
    const std::string body, const std::string name, const std::string build_opt)
{
    const auto ok = _virtual_device->build_task(task, global_size , body, name, build_opt);
    assert(ok == 0);
#if _DEBUG
    std::cout << "Build kernel_function:\t" << name << std::endl;
#endif

    auto kern = task.kernel_cpu();
    kernel_list[kernel_id] = (*kern)();
    _tasks_surf.emplace(kernel_list[kernel_id], &task);
}

template<typename T1,typename T2>
static int execute_two_buffers_func(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size, cl_mem arg0, cl_mem arg1, T1 arg2, T2 arg3)
{
    if (global_work_size == nullptr || local_work_size == nullptr)return -1;

    std::array<size_t, 3> gl_size{ 1,1,1 }, loc_size{ 1,1,1 };

    for (size_t i = 0; i < work_dim; i++) {
        gl_size[i] = global_work_size[i];
        loc_size[i] = local_work_size[i];
    }

    auto cocl_task = _tasks_surf.at(kernel);
    auto cocl_arg0 = &_memory_objects.at(arg0);
    auto cocl_arg1 = &_memory_objects.at(arg1);
    
    if (!cocl_task || !cocl_arg0 || !cocl_arg1 )return -1;
    const auto status = _virtual_device->execute(*cocl_task, _offload, gl_size, loc_size, *cocl_arg0, *cocl_arg1, arg2, arg3);
    assert(status == 0);
#if _DEBUG
    std::cout << "Executed task:\t" << cocl_task->name() << std::endl;
#endif
    return status;
}

int coopcl_execute_scan(cl_kernel kernel, cl_uint work_dim, 
    const size_t* global_work_size, const size_t* local_work_size, cl_mem arg0, cl_mem arg1, int arg2, int arg3)
{
    return execute_two_buffers_func(kernel, work_dim, global_work_size, local_work_size, arg0, arg1, arg2, arg3);
}

int coopcl_execute_transpose(cl_kernel kernel, cl_uint work_dim, 
    const size_t* global_work_size, const size_t* local_work_size, cl_mem arg0, cl_mem arg1, int arg2, int arg3)
{
    return execute_two_buffers_func(kernel, work_dim, global_work_size, local_work_size, arg0, arg1, arg2, arg3);
}

int coopcl_execute_HessianDet(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size, 
    cl_mem arg0, int arg1, int arg2, cl_mem arg3, cl_mem arg4, int arg5, int arg6,int arg7, int arg8)
{
    /*
    cl_setKernelArg(hessian_det, 0, sizeof(cl_mem), (void*)&d_intImage);
    cl_setKernelArg(hessian_det, 1, sizeof(cl_int), (void*)&i_width);
    cl_setKernelArg(hessian_det, 2, sizeof(cl_int), (void*)&i_height);
    cl_setKernelArg(hessian_det, 3, sizeof(cl_mem), (void*)&responses);
    cl_setKernelArg(hessian_det, 4, sizeof(cl_mem), (void*)&laplacian);
    cl_setKernelArg(hessian_det, 5, sizeof(int), (void*)&layerWidth);
    cl_setKernelArg(hessian_det, 6, sizeof(int), (void*)&layerHeight);
    cl_setKernelArg(hessian_det, 7, sizeof(int), (void*)&step);
    cl_setKernelArg(hessian_det, 8, sizeof(int), (void*)&filter);*/
    if (global_work_size == nullptr || local_work_size == nullptr)return -1;

    std::array<size_t, 3> gl_size{ 1,1,1 }, loc_size{ 1,1,1 };

    for (size_t i = 0; i < work_dim; i++) {
        gl_size[i] = global_work_size[i];
        loc_size[i] = local_work_size[i];
    }

    auto cocl_task = _tasks_surf.at(kernel);
    auto cocl_arg0 = &_memory_objects.at(arg0);
    auto cocl_arg3 = &_memory_objects.at(arg3);
    auto cocl_arg4 = &_memory_objects.at(arg4);

    if (!cocl_task || !cocl_arg0 || !cocl_arg3 || !cocl_arg4)return -1;
    
    const auto status = _virtual_device->execute(*cocl_task, _offload, gl_size, loc_size,
        *cocl_arg0,arg1,arg2,*cocl_arg3,*cocl_arg4,arg5,arg6,arg7,arg8);

    assert(status == 0);
#if _DEBUG
    std::cout << "Executed task:\t" << cocl_task->name() << std::endl;
#endif
    return 0;
}

int coopcl_execute_non_max_supression(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size,
    cl_mem arg0,int arg1,int arg2,int arg3,int arg4,cl_mem arg5,cl_mem arg6,
    int arg7,int arg8,int arg9,cl_mem arg10,int arg11,int arg12,int arg13,
    cl_mem arg14, cl_mem arg15, cl_mem arg16, cl_mem arg17, int arg18,float arg19)
{
    //cl_setKernelArg(non_max_supression, 0, sizeof(cl_mem), (void*)&tResponse);
    //cl_setKernelArg(non_max_supression, 1, sizeof(int), (void*)&tWidth);
    //cl_setKernelArg(non_max_supression, 2, sizeof(int), (void*)&tHeight);
    //cl_setKernelArg(non_max_supression, 3, sizeof(int), (void*)&tFilter);
    //cl_setKernelArg(non_max_supression, 4, sizeof(int), (void*)&tStep);
    //cl_setKernelArg(non_max_supression, 5, sizeof(cl_mem), (void*)&mResponse);
    //cl_setKernelArg(non_max_supression, 6, sizeof(cl_mem), (void*)&mLaplacian);
    //cl_setKernelArg(non_max_supression, 7, sizeof(int), (void*)&mWidth);
    //cl_setKernelArg(non_max_supression, 8, sizeof(int), (void*)&mHeight);
    //cl_setKernelArg(non_max_supression, 9, sizeof(int), (void*)&mFilter);
    //cl_setKernelArg(non_max_supression, 10, sizeof(cl_mem), (void*)&bResponse);
    //cl_setKernelArg(non_max_supression, 11, sizeof(int), (void*)&bWidth);
    //cl_setKernelArg(non_max_supression, 12, sizeof(int), (void*)&bHeight);
    //cl_setKernelArg(non_max_supression, 13, sizeof(int), (void*)&bFilter);
    //cl_setKernelArg(non_max_supression, 14, sizeof(cl_mem), (void*)&(this->d_ipt_count));
    //cl_setKernelArg(non_max_supression, 15, sizeof(cl_mem), (void*)&d_pixPos);
    //cl_setKernelArg(non_max_supression, 16, sizeof(cl_mem), (void*)&d_scale);
    //cl_setKernelArg(non_max_supression, 17, sizeof(cl_mem), (void*)&d_laplacian);
    //cl_setKernelArg(non_max_supression, 18, sizeof(int), (void*)&maxPoints);
    //cl_setKernelArg(non_max_supression, 19, sizeof(float), (void*)&(this->thres));
    //// Call non-max supression kernel
    //cl_executeKernel(non_max_supression, 2, globalWorkSize, localWorkSize, "NonMaxSupression", o * 2 + i);
    if (global_work_size == nullptr || local_work_size == nullptr)return -1;

    std::array<size_t, 3> gl_size{ 1,1,1 }, loc_size{ 1,1,1 };

    for (size_t i = 0; i < work_dim; i++) {
        gl_size[i] = global_work_size[i];
        loc_size[i] = local_work_size[i];
    }

    auto cocl_task = _tasks_surf.at(kernel);
    auto cocl_arg0 = &_memory_objects.at(arg0);
    auto cocl_arg5 = &_memory_objects.at(arg5);
    auto cocl_arg6 = &_memory_objects.at(arg6);
    auto cocl_arg10 = &_memory_objects.at(arg10);
    auto cocl_arg14 = &_memory_objects.at(arg14);
    auto cocl_arg15 = &_memory_objects.at(arg15);
    auto cocl_arg16 = &_memory_objects.at(arg16);
    auto cocl_arg17 = &_memory_objects.at(arg17);

    if (!cocl_task || !cocl_arg0 || !cocl_arg5 || !cocl_arg6)return -1;
    if (!cocl_arg10 || !cocl_arg14 || !cocl_arg15 || !cocl_arg16 || !cocl_arg17)return -1;

    const auto status = _virtual_device->execute(*cocl_task, _offload, gl_size, loc_size,
        *cocl_arg0, arg1, arg2, arg3, arg4, *cocl_arg5, *cocl_arg6,
        arg7, arg8, arg9, *cocl_arg10, arg11, arg12, arg13,
        *cocl_arg14, *cocl_arg15, *cocl_arg16, *cocl_arg17, arg18, arg19);

    assert(status == 0);
#if _DEBUG
    std::cout << "Executed task:\t" << cocl_task->name() << std::endl;
#endif
    return 0;

}

int coopcl_execute_orientation1(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size,
    cl_mem arg0, cl_mem arg1, cl_mem arg2, cl_mem arg3, 
    cl_mem arg4, int arg5 , int arg6, cl_mem arg7 )
{
    //cl_setKernelArg(getOrientation, 0, sizeof(cl_mem), (void*)&(this->d_intImage));
    //cl_setKernelArg(getOrientation, 1, sizeof(cl_mem), (void*)&(this->d_scale));
    //cl_setKernelArg(getOrientation, 2, sizeof(cl_mem), (void*)&(this->d_pixPos));
    //cl_setKernelArg(getOrientation, 3, sizeof(cl_mem), (void*)&(this->d_gauss25));
    //cl_setKernelArg(getOrientation, 4, sizeof(cl_mem), (void*)&(this->d_id));
    //cl_setKernelArg(getOrientation, 5, sizeof(int), (void*)&i_width);
    //cl_setKernelArg(getOrientation, 6, sizeof(int), (void*)&i_height);
    //cl_setKernelArg(getOrientation, 7, sizeof(cl_mem), (void*)&(this->d_res));
    //// Execute the kernel
    //cl_executeKernel(getOrientation, 1, globalWorkSize1, localWorkSize1, "GetOrientations");
    if (global_work_size == nullptr || local_work_size == nullptr)return -1;

    std::array<size_t, 3> gl_size{ 1,1,1 }, loc_size{ 1,1,1 };

    for (size_t i = 0; i < work_dim; i++) {
        gl_size[i] = global_work_size[i];
        loc_size[i] = local_work_size[i];
    }

    auto cocl_task = _tasks_surf.at(kernel);
    auto cocl_arg0 = &_memory_objects.at(arg0);
    auto cocl_arg1 = &_memory_objects.at(arg1);
    auto cocl_arg2 = &_memory_objects.at(arg2);
    auto cocl_arg3 = &_memory_objects.at(arg3);
    auto cocl_arg4 = &_memory_objects.at(arg4);
    auto cocl_arg7 = &_memory_objects.at(arg7);

    if (!cocl_task || !cocl_arg0 || !cocl_arg1 || !cocl_arg2 ||!cocl_arg3 || !cocl_arg4 ||!cocl_arg7)return -1;

    const auto status = _virtual_device->execute(*cocl_task, _offload, gl_size, loc_size,
        *cocl_arg0, *cocl_arg1, *cocl_arg2, *cocl_arg3, *cocl_arg4, arg5, arg6, *cocl_arg7);

    assert(status == 0);
#if _DEBUG
    std::cout << "Executed task:\t" << cocl_task->name() << std::endl;
#endif
    return status;
}

int coopcl_execute_orientation2(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size,
    cl_mem arg0, cl_mem arg1)
{
    if (global_work_size == nullptr || local_work_size == nullptr)return -1;

    std::array<size_t, 3> gl_size{ 1,1,1 }, loc_size{ 1,1,1 };

    for (size_t i = 0; i < work_dim; i++) {
        gl_size[i] = global_work_size[i];
        loc_size[i] = local_work_size[i];
    }

    auto cocl_task = _tasks_surf.at(kernel);
    auto cocl_arg0 = &_memory_objects.at(arg0);
    auto cocl_arg1 = &_memory_objects.at(arg1);    

    if (!cocl_task || !cocl_arg0 || !cocl_arg1 )return -1;

    const auto status = _virtual_device->execute(*cocl_task, _offload, gl_size, loc_size,*cocl_arg0, *cocl_arg1);

    assert(status == 0);
#if _DEBUG
    std::cout << "Executed task:\t" << cocl_task->name() << std::endl;
#endif
    return 0;
}

int coopcl_execute_descriptor(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size,
    cl_mem arg0, int arg1, int arg2, cl_mem arg3,cl_mem arg4, 
    cl_mem arg5, cl_mem arg6, cl_mem arg7, cl_mem arg8, cl_mem arg9)
{
    //cl_setKernelArg(surf64Descriptor_kernel, 0, sizeof(cl_mem), (void*)&(this->d_intImage));
    //cl_setKernelArg(surf64Descriptor_kernel, 1, sizeof(int), (void*)&i_width);
    //cl_setKernelArg(surf64Descriptor_kernel, 2, sizeof(int), (void*)&i_height);
    //cl_setKernelArg(surf64Descriptor_kernel, 3, sizeof(cl_mem), (void*)&(this->d_scale));
    //cl_setKernelArg(surf64Descriptor_kernel, 4, sizeof(cl_mem), (void*)&(this->d_desc));
    //cl_setKernelArg(surf64Descriptor_kernel, 5, sizeof(cl_mem), (void*)&(this->d_pixPos));
    //cl_setKernelArg(surf64Descriptor_kernel, 6, sizeof(cl_mem), (void*)&(this->d_orientation));
    //cl_setKernelArg(surf64Descriptor_kernel, 7, sizeof(cl_mem), (void*)&(this->d_length));
    //cl_setKernelArg(surf64Descriptor_kernel, 8, sizeof(cl_mem), (void*)&(this->d_j));
    //cl_setKernelArg(surf64Descriptor_kernel, 9, sizeof(cl_mem), (void*)&(this->d_i));
    //// Execute the descriptor kernel
    //cl_executeKernel(surf64Descriptor_kernel, 2, globalWorkSizeSurf64, localWorkSizeSurf64, "CreateDescriptors");
    if (global_work_size == nullptr || local_work_size == nullptr)return -1;

    std::array<size_t, 3> gl_size{ 1,1,1 }, loc_size{ 1,1,1 };

    for (size_t i = 0; i < work_dim; i++) {
        gl_size[i] = global_work_size[i];
        loc_size[i] = local_work_size[i];
    }

    auto cocl_task = _tasks_surf.at(kernel);
    auto cocl_arg0 = &_memory_objects.at(arg0);    
    auto cocl_arg3 = &_memory_objects.at(arg3);
    auto cocl_arg4 = &_memory_objects.at(arg4);
    auto cocl_arg5 = &_memory_objects.at(arg5);
    auto cocl_arg6 = &_memory_objects.at(arg6);    
    auto cocl_arg7 = &_memory_objects.at(arg7);
    auto cocl_arg8 = &_memory_objects.at(arg8);
    auto cocl_arg9 = &_memory_objects.at(arg9);

    if (!cocl_task || !cocl_arg0 || !cocl_arg3 || !cocl_arg4 || !cocl_arg5)return -1;
    if (!cocl_arg6 || !cocl_arg7 || !cocl_arg8 || !cocl_arg9)return -1;
		
    const auto status = _virtual_device->execute(*cocl_task, _offload, gl_size, loc_size,
        *cocl_arg0, arg1, arg2, *cocl_arg3, *cocl_arg4, *cocl_arg5, *cocl_arg6, *cocl_arg7,*cocl_arg8, *cocl_arg9);

    assert(status == 0);
#if _DEBUG
    std::cout << "Executed task:\t" << cocl_task->name() << std::endl;
#endif
    return status;
}

int coopcl_execute_descriptor_norm(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size,
    cl_mem arg0, cl_mem arg1)
{
    if (global_work_size == nullptr || local_work_size == nullptr)return -1;

    std::array<size_t, 3> gl_size{ 1,1,1 }, loc_size{ 1,1,1 };

    for (size_t i = 0; i < work_dim; i++) {
        gl_size[i] = global_work_size[i];
        loc_size[i] = local_work_size[i];
    }

    auto cocl_task = _tasks_surf.at(kernel);
    auto cocl_arg0 = &_memory_objects.at(arg0);
    auto cocl_arg1 = &_memory_objects.at(arg1);
    
    if (!cocl_task || !cocl_arg0 || !cocl_arg1)return -1;    

    const auto status = _virtual_device->execute(*cocl_task, _offload, gl_size, loc_size,*cocl_arg0, *cocl_arg1);

    assert(status == 0);
#if _DEBUG
    std::cout << "Executed task:\t" << cocl_task->name() << std::endl;
#endif
    return status;
}

#endif

//-------------------------------------------------------
//          Initialization and Cleanup
//-------------------------------------------------------

/*!

    \brief Initialize OpenCl environment on one device

    Init function for one device. Looks for supported devices and creates a context
    \return returns a context initialized
*/
cl_context cl_init(char devicePreference,const float offload)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;

    // Allocate the event table
    events = new EventList();

    // Discover and populate the platforms
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    cl_errChk(status, "Getting platform IDs", true);
    if (numPlatforms > 0)
    {
        // Get all the platforms
        platforms = (cl_platform_id*)alloc(numPlatforms *
            sizeof(cl_platform_id));

        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        cl_errChk(status, "Getting platform IDs", true);
    }
    else
    {
        // If no platforms are available, we shouldn't continue
        printf("No OpenCL platforms found\n");
        exit(-1);
    }

    // Allocate space for the device lists and lengths
    numDevices = (cl_uint*)alloc(sizeof(cl_uint)*numPlatforms);
    devices = (cl_device_id**)alloc(sizeof(cl_device_id*)*numPlatforms);

    // If a device preference was supplied, we'll limit the search of devices
    // based on type
    cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
    if(devicePreference == 'c') {
        deviceType = CL_DEVICE_TYPE_CPU;
    }
    if(devicePreference == 'g') {
        deviceType = CL_DEVICE_TYPE_GPU;
    }

    // Traverse the platforms array printing information and
    // populating devices
    for(unsigned int i = 0; i < numPlatforms ; i++)
    {
        // Print out some basic info about the platform
        char* platformName = NULL;
        char* platformVendor = NULL;

        platformName = cl_getPlatformName(platforms[i]);
        platformVendor = cl_getPlatformVendor(platforms[i]);

        status = clGetDeviceIDs(platforms[i], deviceType, 0, NULL, &numDevices[i]);
        cl_errChk(status, "Getting device IDs", false);
        if(status != CL_SUCCESS) {
            printf("This is a known NVIDIA bug (if platform == AMD then die)\n");
            printf("Setting number of devices to 0 and continuing\n");
            numDevices[i] = 0;
        }

        printf("Platform %d (%d devices):\n", i, numDevices[i]);
        printf("\tName: %s\n", platformName);
        printf("\tVendor: %s\n", platformVendor);

        free(platformName);
        free(platformVendor);

        // Populate OpenCL devices if any exist
        if(numDevices[i] != 0)
        {
            // Allocate an array of devices of size "numDevices"
            devices[i] = (cl_device_id*)alloc(sizeof(cl_device_id)*numDevices[i]);

            // Populate Arrray with devices
            status = clGetDeviceIDs(platforms[i], deviceType, numDevices[i],
                devices[i], NULL);
            cl_errChk(status, "Getting device IDs", true);
        }

        // Print some information about each device
        for( unsigned int j = 0; j < numDevices[i]; j++)
        {
            char* deviceName = NULL;
            char* deviceVendor = NULL;

            printf("\tDevice %d:\n", j);

            deviceName = cl_getDeviceName(devices[i][j]);
            deviceVendor = cl_getDeviceVendor(devices[i][j]);

            printf("\t\tName: %s\n", deviceName);
            printf("\t\tVendor: %s\n", deviceVendor);

            free(deviceName);
            free(deviceVendor);
        }
    }

    // Hard-code in the platform/device to use, or uncomment 'scanf'
    // to decide at runtime
    cl_uint chosen_platform, chosen_device;
    // UNCOMMENT the following two lines to manually select device each time
    printf("Enter Platform and Device No (Seperated by Space) \n");
    scanf("%d %d", &chosen_platform, &chosen_device);
    //chosen_platform = 0;
    //chosen_device = 0;
    printf("Using Platform %d, Device %d \n", chosen_platform, chosen_device);

    // Do a sanity check of platform/device selection
    if(chosen_platform >= numPlatforms ||
        chosen_device >= numDevices[chosen_platform]) {
        printf("Invalid platform/device combination\n");
        exit(-1);
    }

    // Set the selected platform and device
    platform = platforms[chosen_platform];
    device = devices[chosen_platform][chosen_device];

    // Create the context
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platform), 0};
    context = clCreateContext(cps, 1, &device, NULL, NULL, &status);
    cl_errChk(status, "Creating context", true);

    // Create the command queue
    commandQueueProf = clCreateCommandQueue(context, device,
                            CL_QUEUE_PROFILING_ENABLE, &status);
    cl_errChk(status, "creating command queue", true);

    commandQueueNoProf = clCreateCommandQueue(context, device, 0, &status);
    cl_errChk(status, "creating command queue", true);

    if(eventsEnabled) {
        printf("Profiling enabled\n");
        commandQueue = commandQueueProf;
    }
    else {
        printf("Profiling disabled\n");
        commandQueue = commandQueueNoProf;
    }
    return context;
#else
    if(!_virtual_device)
        _virtual_device = std::make_unique<coopcl::virtual_device>();

    _offload = offload;
#endif
    return nullptr;
}

/*!
    Release all resources that the user doesn't have access to.
*/
void  cl_cleanup()
{
#ifdef  _ORG_CL_DRIVER
    // Free the events (this frees the OpenCL events as well)
    delete events;

    // Free the command queue
    if(commandQueue) {
        clReleaseCommandQueue(commandQueue);
    }

    // Free the context
    if(context) {
        clReleaseContext(context);
    }

    // Free the kernel objects
    for(int i = 0; i < NUM_KERNELS; i++) {
        clReleaseKernel(kernel_list[i]);
    }

    // Free the program objects
    for(int i = 0; i < NUM_PROGRAMS; i++) {
        clReleaseProgram(program_list[i]);
    }

    // Free the devices
    for(int i = 0; i < (int)numPlatforms; i++) {
        free(devices[i]);
    }
    free(devices);
    free(numDevices);

    // Free the platforms
    free(platforms);
#else
#endif
    return;
}

//! Release a kernel object
/*!
    \param mem The kernel object to release
*/
void cl_freeKernel(cl_kernel kernel)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;

    if(kernel != NULL) {
        status = clReleaseKernel(kernel);
        cl_errChk(status, "Releasing kernel object", true);
    }
#else
#endif
    return;
}

//! Release memory allocated on the device
/*!
    \param mem The device pointer to release
*/
void cl_freeMem(cl_mem mem)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;

    if(mem != NULL) {
        status = clReleaseMemObject(mem);
        cl_errChk(status, "Releasing mem object", true);
    }
#else
#endif
    return;
}

//! Release a program object
/*!
    \param mem The program object to release
*/
void cl_freeProgram(cl_program program)
{
#ifdef  _ORG_CL_DRIVER
   cl_int status;

    if(program != NULL) {
        status = clReleaseProgram(program);
        cl_errChk(status, "Releasing program object", true);
    }
#else
#endif
    return;
}


//-------------------------------------------------------
//          Synchronization functions
//-------------------------------------------------------

/*!
    Wait till all pending commands in queue are finished
*/
void cl_sync()
{
#ifdef  _ORG_CL_DRIVER
    clFinish(commandQueue);
#else
#endif
    return;
}


//-------------------------------------------------------
//          Memory allocation
//-------------------------------------------------------

//! Allocate a buffer on a device
/*!
    \param mem_size Size of memory in bytes
    \param flags Optional cl_mem_flags
    \return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocBuffer(size_t mem_size, cl_mem_flags flags)
{
#ifdef  _ORG_CL_DRIVER
    cl_mem mem;
    cl_int status;

    /*!
        Logging information for keeping track of device memory
    */
    static int allocationCount = 1;
    static size_t allocationSize = 0;
    allocationCount++;
    allocationSize += mem_size;
    mem = clCreateBuffer(context, flags, mem_size, NULL, &status);
    cl_errChk(status, "creating buffer", true);
    return mem;
#else
    auto coopcl_mem = _virtual_device->alloc<std::uint8_t>(mem_size);
    auto mem_raw = coopcl_mem->get_mem();
    _memory_objects.emplace( mem_raw,std::move(coopcl_mem) );
    return mem_raw;
#endif
    return nullptr;
}

//! Allocate constant memory on device
/*!
    \param mem_size Size of memory in bytes
    \param host_ptr Host pointer that contains the data
    \return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocBufferConst(size_t mem_size, void* host_ptr)
{
#ifdef  _ORG_CL_DRIVER
    cl_mem mem;
    cl_int status;
    mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,mem_size, host_ptr, &status);
    cl_errChk(status, "Error creating const mem buffer", true);
    return mem;
#else    
    auto coopcl_mem = _virtual_device->alloc<std::uint8_t>(mem_size);
    std::memcpy(coopcl_mem->data(), host_ptr, mem_size);
    auto mem_raw = coopcl_mem->get_mem();
    _memory_objects.emplace(mem_raw, std::move(coopcl_mem));
    return mem_raw;
#endif
    return nullptr;
}

//! Allocate a buffer on device pinning the host memory at host_ptr
/*!
    \param mem_size Size of memory in bytes
    \return Returns a cl_mem object that points to pinned memory on the host
*/
cl_mem cl_allocBufferPinned(size_t mem_size)
{
#ifdef  _ORG_CL_DRIVER
    cl_mem mem;
    cl_int status;
    mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,mem_size, NULL, &status);
    cl_errChk(status, "Error allocating pinned memory", true);
    return mem;
#else
    return cl_allocBuffer(mem_size, 0);
#endif
    return nullptr;
}

//! Allocate an image on a device
/*!
    \param height Number of rows in the image
    \param width Number of columns in the image
    \param elemSize Size of the elements in the image
    \param flags Optional cl_mem_flags
    \return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocImage(size_t height, size_t width, char type, cl_mem_flags flags)
{
#ifdef  _ORG_CL_DRIVER
    cl_mem mem;
    cl_int status;
    size_t elemSize = 0;
    cl_image_format format;
    format.image_channel_order = CL_R;

    switch(type) {
    case 'f':
        elemSize = sizeof(float);
        format.image_channel_data_type = CL_FLOAT;
        break;
    case 'i':
        elemSize = sizeof(int);
        format.image_channel_data_type = CL_SIGNED_INT32;
        break;
    default:
        printf("Error creating image: Unsupported image type.\n");
        exit(-1);
    }

    /*!
        Logging information for keeping track of device memory
    */
    static int allocationCount = 1;
    static size_t allocationSize = 0;

    allocationCount++;
    allocationSize += height*width*elemSize;

    // Create the image
    mem = clCreateImage2D(context, flags, &format, width, height, 0, NULL, &status);

    //cl_errChk(status, "creating image", true);
    if(status != CL_SUCCESS) {
        printf("Error creating image: Images may not be supported for this device.\n");
        printSupportedImageFormats();
        getchar();
        exit(-1);
    }

    return mem;
#else
#endif
    return nullptr;
}


//-------------------------------------------------------
//          Data transfers
//-------------------------------------------------------


// Copy and map a buffer
void* cl_copyAndMapBuffer(cl_mem dst, cl_mem src, size_t size) 
{
#ifdef  _ORG_CL_DRIVER
    void* ptr;  // Pointer to the pinned memory that will be returned
    cl_copyBufferToBuffer(dst, src, size);
    ptr = cl_mapBuffer(dst, size, CL_MAP_READ);
    return ptr;
#else
#endif
    return nullptr;
}

// Copy a buffer
void cl_copyBufferToBuffer(cl_mem dst, cl_mem src, size_t size)
{
#ifdef  _ORG_CL_DRIVER
    static int eventCnt = 0;
    cl_event* eventPtr = NULL, event;

    if(eventsEnabled) {
        eventPtr = &event;
    }

    cl_int status;
    status = clEnqueueCopyBuffer(commandQueue, src, dst, 0, 0, size, 0, NULL,
        eventPtr);
    cl_errChk(status, "Copying buffer", true);

    if(eventsEnabled) {
        char* eventStr = catStringWithInt("copyBuffer", eventCnt++);
        events->newIOEvent(*eventPtr, eventStr);
    }
#else

#endif
    return;
}

//! Copy a buffer to the device
/*!
    \param dst Valid device pointer
    \param src Host pointer that contains the data
    \param mem_size Size of data to copy
	\param blocking Blocking or non-blocking operation
*/
void cl_copyBufferToDevice(cl_mem dst, void* src, size_t mem_size, cl_bool blocking)
{
#ifdef  _ORG_CL_DRIVER
    static int eventCnt = 0;
    cl_event* eventPtr = NULL, event;

    if(eventsEnabled) {
        eventPtr = &event;
    }

    cl_int status;
    status = clEnqueueWriteBuffer(commandQueue, dst, blocking, 0,
        mem_size, src, 0, NULL, eventPtr);
    cl_errChk(status, "Writing buffer", true);

    if(eventsEnabled) {
        char* eventStr = catStringWithInt("copyBufferToDevice", eventCnt++);
        events->newIOEvent(*eventPtr, eventStr);
    }
#else
    //get memory_obj and copy from src
    auto& coopcl_mem = _memory_objects.at(dst);
    std::memcpy(coopcl_mem->data(), src, mem_size);
#endif       
    return;
}

//! Copy a buffer to the host
/*!
    \param dst Valid host pointer
    \param src Device pointer that contains the data
    \param mem_size Size of data to copy
	\param blocking Blocking or non-blocking operation
*/
void cl_copyBufferToHost(void* dst, cl_mem src, size_t mem_size, cl_bool blocking)
{
#ifdef  _ORG_CL_DRIVER
    static int eventCnt = 0;

    cl_event* eventPtr = NULL, event;

    if(eventsEnabled) {
        eventPtr = &event;
    }

    cl_int status;
    status = clEnqueueReadBuffer(commandQueue, src, blocking, 0,
        mem_size, dst, 0, NULL, eventPtr);
    cl_errChk(status, "Reading buffer", true);

    if(eventsEnabled) {
        char* eventStr = catStringWithInt("copyBufferToHost", eventCnt++);
        events->newIOEvent(*eventPtr, eventStr);
    }
#else
    //get memory_obj and copy from src
    auto& coopcl_mem = _memory_objects.at(src);
    std::memcpy(dst, coopcl_mem->data(), mem_size);
#endif
    return;
}

//! Copy a buffer to a 2D image
/*!
    \param src Valid device buffer
    \param dst Empty device image
    \param mem_size Size of data to copy
*/
void cl_copyBufferToImage(cl_mem buffer, cl_mem image, int height, int width)
{
#ifdef  _ORG_CL_DRIVER
    static int eventCnt = 0;

    cl_event* eventPtr = NULL, event;

    if(eventsEnabled) {
        eventPtr = &event;
    }

    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    cl_int status;
    status = clEnqueueCopyBufferToImage(commandQueue, buffer, image, 0,
        origin, region, 0, NULL, eventPtr);
    cl_errChk(status, "Copying buffer to image", true);

    if(eventsEnabled) {
        char* eventStr = catStringWithInt("copyBufferToImage", eventCnt++);
        events->newIOEvent(*eventPtr, eventStr);
    }
#else
#endif
    return;
}

// Copy data to an image on the device
/*!
    \param dst Valid device pointer
    \param src Host pointer that contains the data
    \param height Height of the image
    \param width Width of the image
*/
void cl_copyImageToDevice(cl_mem dst, void* src, size_t height, size_t width)
{
#ifdef  _ORG_CL_DRIVER
    static int eventCnt = 0;

    cl_event* eventPtr = NULL, event;

    if(eventsEnabled) {
        eventPtr = &event;
    }

    cl_int status;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    status = clEnqueueWriteImage(commandQueue, dst, CL_TRUE, origin,
        region, 0, 0, src, 0, NULL, eventPtr);
    cl_errChk(status, "Writing image", true);

    if(eventsEnabled) {
        char* eventStr = catStringWithInt("copyImageToDevice", eventCnt++);
        events->newIOEvent(*eventPtr, eventStr);
    }
#else
#endif
    return;
}

//! Copy an image to the host
/*!
    \param dst Valid host pointer
    \param src Device pointer that contains the data
    \param height Height of the image
    \param width Width of the image
*/
void cl_copyImageToHost(void* dst, cl_mem src, size_t height, size_t width)
{
#ifdef  _ORG_CL_DRIVER
    static int eventCnt = 0;

    cl_event* eventPtr = NULL, event;

    if(eventsEnabled) {
        eventPtr = &event;
    }

    cl_int status;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    status = clEnqueueReadImage(commandQueue, src, CL_TRUE, origin,
        region, 0, 0, dst, 0, NULL, eventPtr);
    cl_errChk(status, "Reading image", true);

    if(eventsEnabled) {
        char* eventStr = catStringWithInt("copyImageToHost", eventCnt++);
        events->newIOEvent(*eventPtr, eventStr);
    }
#else
#endif
    return;
}

//! Map a buffer into a host address
/*!
    \param mem cl_mem object
	\param mem_size Size of memory in bytes
    \param flags Optional cl_mem_flags
    \return Returns a host pointer that points to the mapped region
*/
void *cl_mapBuffer(cl_mem mem, size_t mem_size, cl_mem_flags flags)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;
    void *ptr;

    static int eventCnt = 0;

    cl_event* eventPtr = NULL, event;

    if(eventsEnabled) {
        eventPtr = &event;
    }

    ptr = (void *)clEnqueueMapBuffer(commandQueue, mem, CL_TRUE, flags,
		                             0, mem_size, 0, NULL, eventPtr, &status);

    cl_errChk(status, "Error mapping a buffer", true);

    if(eventsEnabled) {
        char* eventStr = catStringWithInt("MapBuffer", eventCnt++);
        events->newIOEvent(*eventPtr, eventStr);
    }

    return ptr;
#else
#endif
    return nullptr;
}

//! Unmap a buffer or image
/*!
    \param mem cl_mem object
    \param ptr A host pointer that points to the mapped region
*/
void cl_unmapBuffer(cl_mem mem, void *ptr)
{
#ifdef  _ORG_CL_DRIVER
    // TODO It looks like AMD doesn't support profiling unmapping yet. Leaving the
    //      commented code here until it's supported

    cl_int status;
    status = clEnqueueUnmapMemObject(commandQueue, mem, ptr, 0, NULL, NULL);
    cl_errChk(status, "Error unmapping a buffer or image", true);
#else
#endif
    return;
}

void cl_writeToZCBuffer(cl_mem mem, void* data, size_t size)
{
#ifdef  _ORG_CL_DRIVER
    void* ptr;
    ptr = cl_mapBuffer(mem, size, CL_MAP_WRITE);
    memcpy(ptr, data, size);
    cl_unmapBuffer(mem, ptr);
#else
#endif
    return;
}

//-------------------------------------------------------
//          Program and kernels
//-------------------------------------------------------

//! Convert source code file into cl_program
/*!
Compile Opencl source file into a cl_program. The cl_program will be made into a kernel in PrecompileKernels()

\param kernelPath  Filename of OpenCl code
\param compileoptions Compilation options
\param verbosebuild Switch to enable verbose Output
*/
cl_program cl_compileProgram(char* kernelPath, char* compileoptions, bool verbosebuild )
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;
    FILE *fp = NULL;
    char *source = NULL;
    long int size;

    printf("\t%s\n", kernelPath);

    // Determine the size of the source file
#ifdef _WIN32
    fopen_s(&fp, kernelPath, "rb");
#else
    fp = fopen(kernelPath, "rb");
#endif
    if(!fp) {
        printf("Could not open kernel file\n");
        exit(-1);
    }
    status = fseek(fp, 0, SEEK_END);
    if(status != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    size = ftell(fp);
    if(size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }
    rewind(fp);

    // Allocate enough space for the source code
    source = (char *)alloc(size + 1);

    // fill with NULLs (just for fun)
    for (int i = 0; i < size+1; i++)  {
        source[i] = '\0';
    }

    // Read in the source code
    fread(source, 1, size, fp);
    source[size] = '\0';

    // Create the program object
    cl_program clProgramReturn = clCreateProgramWithSource(context, 1,
        (const char **)&source, NULL, &status);
    cl_errChk(status, "Creating program", true);

    free(source);
    fclose(fp);

    // Try to compile the program
    status = clBuildProgram(clProgramReturn, 0, NULL, compileoptions, NULL, NULL);
    if(cl_errChk(status, "Building program", false) || verbosebuild == 1)
    {

        cl_build_status build_status;

        clGetProgramBuildInfo(clProgramReturn, device, CL_PROGRAM_BUILD_STATUS,
            sizeof(cl_build_status), &build_status, NULL);

        if(build_status == CL_SUCCESS && verbosebuild == 0) {
            return clProgramReturn;
        }

        //char *build_log;
        size_t ret_val_size;
        printf("Device: %p",device);
        clGetProgramBuildInfo(clProgramReturn, device, CL_PROGRAM_BUILD_LOG, 0,
            NULL, &ret_val_size);

        char *build_log = (char*)alloc(ret_val_size+1);

        clGetProgramBuildInfo(clProgramReturn, device, CL_PROGRAM_BUILD_LOG,
            ret_val_size+1, build_log, NULL);

        // to be careful, terminate with \0
        // there's no information in the reference whether the string is 0
        // terminated or not
        build_log[ret_val_size] = '\0';

        printf("Build log:\n %s...\n", build_log);
        if(build_status != CL_SUCCESS) {
            getchar();
            exit(-1);
        }
        else
            return clProgramReturn;
    }

    // print the ptx information
    // printBinaries(clProgram);

    return clProgramReturn;
#else
#endif
    return nullptr;
}

//! Create a kernel from compiled source
/*!
Create a kernel from compiled source

\param program  Compiled OpenCL program
\param kernel_name  Name of the kernel in the program
\return Returns a cl_kernel object for the specified kernel
*/
cl_kernel cl_createKernel(cl_program program, const char* kernel_name) 
{
#ifdef  _ORG_CL_DRIVER
    cl_kernel kernel;
    cl_int status;
    kernel = clCreateKernel(program, kernel_name, &status);
    cl_errChk(status, "Creating kernel", true);
    return kernel;
#else
#endif
    return nullptr;
}

//! Enqueue and NDRange kernel on a device
/*!
    \param kernel The kernel to execute
    \param work_dim  The number of dimensions that define the thread structure
    \param global_work_size  Array of size 'work_dim' that defines the total threads in each dimension
    \param local_work_size  Array of size 'work_dim' that defines the size of each work group
    \param description String describing the kernel
    \param identifier A number unique number identifying the kernel
*/
int global_event_ctr = 0;

void cl_executeKernel(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size,
    const char* description, int identifier)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;
    cl_event* eventPtr = NULL, event;
//    eventsEnabled =  phasechecker(description, identifier, granularity);
    if(eventsEnabled) {
        eventPtr = &event;
    }
    status = clEnqueueNDRangeKernel(commandQueue, kernel, work_dim, NULL,
        global_work_size, local_work_size, 0, NULL, eventPtr);
    cl_errChk(status, "Executing kernel", true);


    if(eventsEnabled) {
        char* eventString = catStringWithInt(description, identifier);
        events->newKernelEvent(*eventPtr, eventString);
    }
#else   
#endif
    return;
}

//! SURF specific kernel precompilation call
/*!
*/
cl_kernel* cl_precompileKernels(char* buildOptions, const std::string kpath,
    const std::array<size_t,3>* global_size)
{
#ifdef  _ORG_CL_DRIVER
    // Compile each program and create the kernel objects

    printf("Precompiling kernels...\n");

    cl_time totalstart, totalend;
    cl_time start, end;

    std::string path;
#ifdef WIN32
    //path = "C:/Development/SequenceStreamDemo/SURF/coopcl_surf/CLSource/";
    path = kpath;
#else
    path = "/home/SequencStreamDemo/clSURF/CLSource/";
#endif

    cl_getTime(&totalstart);

    // Creating descriptors kernel
    cl_getTime(&start);    
	std::string k1_cl = path; k1_cl.append("createDescriptors_kernel.cl");
	program_list[1]  = cl_compileProgram((char*)k1_cl.data(),buildOptions, false);
    cl_getTime(&end);
    events->newCompileEvent(cl_computeTime(start, end), "createDescriptors");
    kernel_list[KERNEL_SURF_DESC] = cl_createKernel(program_list[1],"createDescriptors");

        // Get orientation kernels
    cl_getTime(&start);    
	std::string k2_cl = path; k2_cl.append("getOrientation_kernels.cl");
	program_list[4] = cl_compileProgram((char*)k2_cl.data(), buildOptions, false);
    cl_getTime(&end);
    events->newCompileEvent(cl_computeTime(start, end), "Orientation");
    kernel_list[KERNEL_GET_ORIENT1] = cl_createKernel(program_list[4],"getOrientationStep1");
    kernel_list[KERNEL_GET_ORIENT2] = cl_createKernel(program_list[4],"getOrientationStep2");

    // Hessian determinant kernel
    cl_getTime(&start);    
	std::string k3_cl = path; k3_cl.append("hessianDet_kernel.cl");
	program_list[0] = cl_compileProgram((char*)k3_cl.data(), buildOptions, false);
    cl_getTime(&end);
    events->newCompileEvent(cl_computeTime(start, end), "hessian_det");
    kernel_list[KERNEL_BUILD_DET] = cl_createKernel(program_list[0],"hessian_det");

    // Integral image kernels
    cl_getTime(&start);    
	std::string k4_cl = path; k4_cl.append("integralImage_kernels.cl");
	program_list[6] = cl_compileProgram((char*)k4_cl.data(), buildOptions, false);
    cl_getTime(&end);
    events->newCompileEvent(cl_computeTime(start, end), "IntegralImage");
    kernel_list[KERNEL_SCAN] = cl_createKernel(program_list[6], "scan");
    kernel_list[KERNEL_SCAN4] = cl_createKernel(program_list[6], "scan4");
    kernel_list[KERNEL_SCANIMAGE] = cl_createKernel(program_list[6],"scanImage");
    kernel_list[KERNEL_TRANSPOSE] = cl_createKernel(program_list[6],"transpose");
    kernel_list[KERNEL_TRANSPOSEIMAGE] = cl_createKernel(program_list[6],"transposeImage");

    // Nearest neighbor kernels
    cl_getTime(&start);    
	std::string k5_cl = path; k5_cl.append("nearestNeighbor_kernel.cl");
	program_list[5] = cl_compileProgram((char*)k5_cl.data(), buildOptions, false);
    cl_getTime(&end);
    events->newCompileEvent(cl_computeTime(start, end), "NearestNeighbor");
    kernel_list[KERNEL_NN] = cl_createKernel(program_list[5],"NearestNeighbor");

    // Non-maximum suppression kernel
    cl_getTime(&start);    
	std::string k6_cl = path; k6_cl.append("nonMaxSuppression_kernel.cl");
	program_list[3] = cl_compileProgram((char*)k6_cl.data(), buildOptions, false);
    cl_getTime(&end);
    events->newCompileEvent(cl_computeTime(start, end), "NonMaxSuppression");
    kernel_list[KERNEL_NON_MAX_SUP] = cl_createKernel(program_list[3],"non_max_supression");

    // Normalization of descriptors kernel
    cl_getTime(&start);    
	std::string k7_cl = path; k7_cl.append("normalizeDescriptors_kernel.cl");
	program_list[2] = cl_compileProgram((char*)k7_cl.data(), buildOptions, false);
    cl_getTime(&end);
    events->newCompileEvent(cl_computeTime(start, end), "normalize");
    kernel_list[KERNEL_NORM_DESC] = cl_createKernel(program_list[2],"normalizeDescriptors");

    cl_getTime(&totalend);

    printf("\tTime for Off-Critical Path Compilation: %.3f milliseconds\n\n",
        cl_computeTime(totalstart, totalend));

    return kernel_list;
#else
    auto read_file = [](const std::string& file_path, std::string& out)->bool {
    std::ifstream ifs(file_path);
    if (!ifs.is_open())return false;
    out.clear();
    std::string line;
    while (getline(ifs, line))
    {
        out.append(line);
        out.append("\n");
    }
    return false;
};
    
    std::string body, name, build_opt;
    build_opt = buildOptions == nullptr ? "" : buildOptions;
    //--------------------------------------------
    // Creating descriptors kernel
	std::string k1_cl = kpath; k1_cl.append("createDescriptors_kernel.cl");
    read_file(k1_cl, body);
    name = "createDescriptors";        
    auto gs_desc = *global_size;
    gs_desc[0] = DESC_TH_WGS * DESC_WGS_ITEM;
    gs_desc[1] = INIT_FEATS;
	task_descriptor_surf.set_ndr_dim_to_divide(1);
    build_kernel_task_map(task_descriptor_surf, KERNEL_SURF_DESC, gs_desc, body, name, build_opt);
    //kernel_list[KERNEL_SURF_DESC] = cl_createKernel(program_list[1],"createDescriptors_kernel");
    //--------------------------------------------
    // Get orientation kernels
	std::string k2_cl = kpath; k2_cl.append("getOrientation_kernels.cl");
    read_file(k2_cl, body);
    name = "getOrientationStep1";    
    auto gs_orient1 = *global_size;
    gs_orient1[0] = ORIENT_WGS_STEP1 * INIT_FEATS;
    build_kernel_task_map(task_orient1_surf, KERNEL_GET_ORIENT1, gs_orient1, body, name, build_opt);
    //kernel_list[KERNEL_GET_ORIENT1] = cl_createKernel(program_list[4],"getOrientationStep1");
    name = "getOrientationStep2";    
    auto gs_orient2 = *global_size;
    gs_orient2[0] = ORIENT_WGS_STEP2 * INIT_FEATS;
    build_kernel_task_map(task_orient2_surf, KERNEL_GET_ORIENT2, gs_orient2, body, name, build_opt);
    //kernel_list[KERNEL_GET_ORIENT2] = cl_createKernel(program_list[4],"getOrientationStep2");   
    //--------------------------------------------
    // Hessian determinant kernel
	std::string k3_cl = kpath; k3_cl.append("hessianDet_kernel.cl");
    read_file(k3_cl, body);
    name = "hessian_det";       
    build_kernel_task_map(task_hessianDet_surf, KERNEL_BUILD_DET, *global_size, body, name, build_opt);
    //kernel_list[KERNEL_BUILD_DET] = cl_createKernel(program_list[0],"hessian_det");    
    //--------------------------------------------
    //Integral image kernels
    std::string k4_cl = kpath; k4_cl.append("integralImage_kernels.cl");
    read_file(k4_cl, body);
    name = "scan";    
	task_scan_surf.set_ndr_dim_to_divide(1);
    build_kernel_task_map(task_scan_surf, KERNEL_SCAN, *global_size, body, name, build_opt);
    //kernel_list[KERNEL_SCAN] = cl_createKernel(program_list[6], "scan");
    //--------------------------------------------
    name = "scan4";    
    build_kernel_task_map(task_scan4_surf, KERNEL_SCAN4, *global_size, body, name, build_opt);
    //kernel_list[KERNEL_SCAN4] = cl_createKernel(program_list[6], "scan4");
    //--------------------------------------------
    name = "scanImage";    
    build_kernel_task_map(task_scanImage_surf, KERNEL_SCANIMAGE, *global_size, body, name, build_opt);
    //kernel_list[KERNEL_SCANIMAGE] = cl_createKernel(program_list[6],"scanImage");
    //--------------------------------------------
    name = "transpose";    
    build_kernel_task_map(task_transpose_surf, KERNEL_TRANSPOSE, *global_size, body, name, build_opt);
    //kernel_list[KERNEL_TRANSPOSE] = cl_createKernel(program_list[6],"transpose");
    //--------------------------------------------
    name = "transposeImage";    
    build_kernel_task_map(task_transposeImage_surf, KERNEL_TRANSPOSEIMAGE, *global_size, body, name, build_opt);
    //kernel_list[KERNEL_TRANSPOSEIMAGE] = cl_createKernel(program_list[6],"transposeImage");
    //--------------------------------------------
    // Nearest neighbor kernels
	std::string k5_cl = kpath; k5_cl.append("nearestNeighbor_kernel.cl");
    read_file(k5_cl, body);
    name = "NearestNeighbor";        
    build_kernel_task_map(task_NearestNeighbor_surf, KERNEL_NN, *global_size, body, name, build_opt);
    //kernel_list[KERNEL_NN] = cl_createKernel(program_list[5],"NearestNeighbor");
    //--------------------------------------------
    // Non-maximum suppression kernel
    std::string k6_cl = kpath; k6_cl.append("nonMaxSuppression_kernel.cl");
    read_file(k6_cl, body);
    name = "non_max_supression";      
    build_kernel_task_map(task_non_max_supression_surf, KERNEL_NON_MAX_SUP, *global_size, body, name, build_opt);
    //kernel_list[KERNEL_NON_MAX_SUP] = cl_createKernel(program_list[3],"non_max_supression_kernel");
    //--------------------------------------------
    // Normalization of descriptors kernel
	std::string k7_cl = kpath; k7_cl.append("normalizeDescriptors_kernel.cl");
    read_file(k7_cl, body);
    name = "normalizeDescriptors";    
    auto gs_desc_norm = *global_size;
    gs_desc_norm[0] = DESC_SIZE * INIT_FEATS;
    build_kernel_task_map(task_normalizeDescriptors_surf, KERNEL_NORM_DESC, gs_desc_norm, body, name, build_opt);
    //kernel_list[KERNEL_NORM_DESC] = cl_createKernel(program_list[2],"normalizeDescriptors");
    return kernel_list;
#endif
    return nullptr;
}

//! Set an argument for a OpenCL kernel
/*!
Set an argument for a OpenCL kernel

\param kernel The kernel for which the argument is being set
\param index The argument index
\param size The size of the argument
\param data A pointer to the argument
*/
void cl_setKernelArg(cl_kernel kernel, unsigned int index, size_t size,
                     void* data)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;
    status = clSetKernelArg(kernel, index, size, data);
    cl_errChk(status, "Setting kernel arg", true);
#else

#endif
    return;
}


//-------------------------------------------------------
//          Profiling/events
//-------------------------------------------------------


//! Time kernel execution using cl_event
/*!
    Prints out the time taken between the start and end of an event
    \param event_time
*/
double cl_computeExecTime(cl_event event_time)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;
    cl_ulong starttime;
    cl_ulong endtime;

    double elapsed=0;

    status = clGetEventProfilingInfo(event_time, CL_PROFILING_COMMAND_START,
                                          sizeof(cl_ulong), &starttime, NULL);
    cl_errChk(status, "profiling start", true);

    status = clGetEventProfilingInfo(event_time, CL_PROFILING_COMMAND_END,
                                          sizeof(cl_ulong), &endtime, NULL);
    cl_errChk(status, "profiling end", true);
    // Convert to ms
    elapsed = (double)(endtime-starttime)/1000000.0;
    return elapsed;
#else
    return 0;
#endif    
}

//! Compute the elapsed time between two timer values
double cl_computeTime(cl_time start, cl_time end)
{
#ifdef  _ORG_CL_DRIVER
#ifdef _WIN32
    __int64 freq;
    int status;

    status = QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    if(status == 0) {
        perror("QueryPerformanceFrequency");
        exit(-1);
    }

    // Return time in ms
    return double(end-start)/(double(freq)/1000.0);
#else

    return end-start;
#endif
#else
    return 0;
#endif
}

//! Create a new user event
void cl_createUserEvent(cl_time start, cl_time end, char* desc) 
{
#ifdef  _ORG_CL_DRIVER
    if(!eventsEnabled) {
        return;
    }

    events->newUserEvent(cl_computeTime(start, end), desc);
#else
#endif
    return;
}

//! Disables events
void cl_disableEvents() 
{
#ifdef  _ORG_CL_DRIVER
    commandQueue = commandQueueNoProf;
    eventsEnabled = false;
    printf("Profiling disabled\n");
#else
#endif
    return;
}

//! Enables events
void cl_enableEvents() 
{
#ifdef  _ORG_CL_DRIVER
    commandQueue = commandQueueProf;
    eventsEnabled = true;
    printf("Profiling enabled\n");
#else
#endif
    return;
}

//! Grab the current time using a system-specific timer
void cl_getTime(cl_time* time)
{
#ifdef  _ORG_CL_DRIVER
#ifdef _WIN32
    int status = QueryPerformanceCounter((LARGE_INTEGER*)time);
    if(status == 0) {
        perror("QueryPerformanceCounter");
        exit(-1);
    }
#else
    // Use gettimeofday to get the current time
    struct timeval curTime;
    gettimeofday(&curTime, NULL);

    // Convert timeval into double
    *time = curTime.tv_sec * 1000 + (double)curTime.tv_usec/1000;
#endif
#else
#endif
    return;
}

//! Print out the OpenCL events
void cl_printEvents() 
{
#ifdef  _ORG_CL_DRIVER
    events->printAllExecTimes();
#else
#endif
    return;
}

//! Write out all current events to a file
void cl_writeEventsToFile(char* path) 
{
#ifdef  _ORG_CL_DRIVER
    events->dumpCSV(path);
    events->dumpTraceCSV(path);
#else
#endif
    return;
}


//-------------------------------------------------------
//          Error handling
//-------------------------------------------------------

//! OpenCl error code list
/*!
    An array of character strings used to give the error corresponding to the error code \n

    The error code is the index within this array
*/
char *cl_errs[MAX_ERR_VAL] = {
    "CL_SUCCESS",                         // 0
    "CL_DEVICE_NOT_FOUND",                //-1
    "CL_DEVICE_NOT_AVAILABLE",            //-2
    "CL_COMPILER_NOT_AVAILABLE",          //-3
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",   //-4
    "CL_OUT_OF_RESOURCES",                //-5
    "CL_OUT_OF_HOST_MEMORY",              //-6
    "CL_PROFILING_INFO_NOT_AVAILABLE",    //-7
    "CL_MEM_COPY_OVERLAP",                //-8
    "CL_IMAGE_FORMAT_MISMATCH",           //-9
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",      //-10
    "CL_BUILD_PROGRAM_FAILURE",           //-11
    "CL_MAP_FAILURE",                     //-12
    "",                                   //-13
    "",                                   //-14
    "",                                   //-15
    "",                                   //-16
    "",                                   //-17
    "",                                   //-18
    "",                                   //-19
    "",                                   //-20
    "",                                   //-21
    "",                                   //-22
    "",                                   //-23
    "",                                   //-24
    "",                                   //-25
    "",                                   //-26
    "",                                   //-27
    "",                                   //-28
    "",                                   //-29
    "CL_INVALID_VALUE",                   //-30
    "CL_INVALID_DEVICE_TYPE",             //-31
    "CL_INVALID_PLATFORM",                //-32
    "CL_INVALID_DEVICE",                  //-33
    "CL_INVALID_CONTEXT",                 //-34
    "CL_INVALID_QUEUE_PROPERTIES",        //-35
    "CL_INVALID_COMMAND_QUEUE",           //-36
    "CL_INVALID_HOST_PTR",                //-37
    "CL_INVALID_MEM_OBJECT",              //-38
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", //-39
    "CL_INVALID_IMAGE_SIZE",              //-40
    "CL_INVALID_SAMPLER",                 //-41
    "CL_INVALID_BINARY",                  //-42
    "CL_INVALID_BUILD_OPTIONS",           //-43
    "CL_INVALID_PROGRAM",                 //-44
    "CL_INVALID_PROGRAM_EXECUTABLE",      //-45
    "CL_INVALID_KERNEL_NAME",             //-46
    "CL_INVALID_KERNEL_DEFINITION",       //-47
    "CL_INVALID_KERNEL",                  //-48
    "CL_INVALID_ARG_INDEX",               //-49
    "CL_INVALID_ARG_VALUE",               //-50
    "CL_INVALID_ARG_SIZE",                //-51
    "CL_INVALID_KERNEL_ARGS",             //-52
    "CL_INVALID_WORK_DIMENSION ",         //-53
    "CL_INVALID_WORK_GROUP_SIZE",         //-54
    "CL_INVALID_WORK_ITEM_SIZE",          //-55
    "CL_INVALID_GLOBAL_OFFSET",           //-56
    "CL_INVALID_EVENT_WAIT_LIST",         //-57
    "CL_INVALID_EVENT",                   //-58
    "CL_INVALID_OPERATION",               //-59
    "CL_INVALID_GL_OBJECT",               //-60
    "CL_INVALID_BUFFER_SIZE",             //-61
    "CL_INVALID_MIP_LEVEL",               //-62
    "CL_INVALID_GLOBAL_WORK_SIZE"};       //-63

//! OpenCl Error checker
/*!
Checks for error code as per cl_int returned by OpenCl
\param status Error value as cl_int
\param msg User provided error message
\return True if Error Seen, False if no error
*/
int cl_errChk(const cl_int status, const char * msg, bool exitOnErr)
{
    if(status != CL_SUCCESS) {
        printf("OpenCL Error: %d %s %s\n", status, cl_errs[-status], msg);

        if(exitOnErr) {
            exit(-1);
        }

        return true;
    }
    return false;
}

// Queries the supported image formats for the device and prints
// them to the screen
 void printSupportedImageFormats()
{
#ifdef  _ORG_CL_DRIVER
    cl_uint numFormats;
    cl_int status;

    status = clGetSupportedImageFormats(context, 0, CL_MEM_OBJECT_IMAGE2D,
        0, NULL, &numFormats);
    cl_errChk(status, "getting supported image formats", true);

    cl_image_format* imageFormats = NULL;
    imageFormats = (cl_image_format*)alloc(sizeof(cl_image_format)*numFormats);

    status = clGetSupportedImageFormats(context, 0, CL_MEM_OBJECT_IMAGE2D,
        numFormats, imageFormats, NULL);

    printf("There are %d supported image formats\n", numFormats);

    cl_uint orders[]={CL_R,  CL_A, CL_INTENSITY, CL_LUMINANCE, CL_RG,
        CL_RA, CL_RGB, CL_RGBA, CL_ARGB, CL_BGRA};
    char  *orderstr[]={"CL_R", "CL_A","CL_INTENSITY", "CL_LUMINANCE", "CL_RG",
        "CL_RA", "CL_RGB", "CL_RGBA", "CL_ARGB", "CL_BGRA"};

    cl_uint types[]={
        CL_SNORM_INT8 , CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
        CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010,CL_SIGNED_INT8,
        CL_SIGNED_INT16,  CL_SIGNED_INT32, CL_UNSIGNED_INT8, CL_UNSIGNED_INT16,
        CL_UNSIGNED_INT32, CL_HALF_FLOAT, CL_FLOAT};

    char * typesstr[]={
        "CL_SNORM_INT8" ,"CL_SNORM_INT16","CL_UNORM_INT8","CL_UNORM_INT16",
        "CL_UNORM_SHORT_565","CL_UNORM_SHORT_555","CL_UNORM_INT_101010",
        "CL_SIGNED_INT8","CL_SIGNED_INT16","CL_SIGNED_INT32","CL_UNSIGNED_INT8",
        "CL_UNSIGNED_INT16","CL_UNSIGNED_INT32","CL_HALF_FLOAT","CL_FLOAT"};

    printf("Supported Formats:\n");
    for(int i = 0; i < (int)numFormats; i++) {
        printf("\tFormat %d: ", i);

        for(int j = 0; j < (int)(sizeof(orders)/sizeof(cl_int)); j++) {
            if(imageFormats[i].image_channel_order == orders[j]) {
                printf("%s, ", orderstr[j]);
            }
        }
        for(int j = 0; j < (int)(sizeof(types)/sizeof(cl_int)); j++) {
            if(imageFormats[i].image_channel_data_type == types[j]) {
                printf("%s, ", typesstr[j]);
            }
        }
        printf("\n");
    }

    free(imageFormats);
#else
#endif
     return;
}


//-------------------------------------------------------
//          Platform and device information
//-------------------------------------------------------

//! Returns true if AMD is the device vendor
bool cl_deviceIsAMD(cl_device_id dev) 
{
    bool retval = false;
#ifdef  _ORG_CL_DRIVER
    char* vendor = cl_getDeviceVendor(dev);

    if(strncmp(vendor, "Advanced", 8) == 0) {
        retval = true;
    }

    free(vendor);
#else
#endif
    return retval;
}

//! Returns true if NVIDIA is the device vendor
bool cl_deviceIsNVIDIA(cl_device_id dev) 
{
    bool retval = false;
#ifdef  _ORG_CL_DRIVER
    char* vendor = cl_getDeviceVendor(dev);

    if(strncmp(vendor, "NVIDIA", 6) == 0) {
        retval = true;
    }

    free(vendor);
#else
#endif

    return retval;
}

//! Returns true if NVIDIA is the device vendor
bool cl_platformIsNVIDIA(cl_platform_id plat) 
{
    bool retval = false;
#ifdef  _ORG_CL_DRIVER
    char* vendor = cl_getPlatformVendor(plat);
    if(strncmp(vendor, "NVIDIA", 6) == 0) {
        retval = true;
    }
    free(vendor);
#else
#endif
    return retval;
}

//! Get the name of the vendor for a device
char* cl_getDeviceDriverVersion(cl_device_id dev)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;
    size_t devInfoSize;
    char* devInfoStr = NULL;

    // If dev is NULL, set it to the default device
    if(dev == NULL) {
        dev = device;
    }

    // Print the vendor
    status = clGetDeviceInfo(dev, CL_DRIVER_VERSION, 0,
        NULL, &devInfoSize);
    cl_errChk(status, "Getting vendor name", true);

    devInfoStr = (char*)alloc(devInfoSize);

    status = clGetDeviceInfo(dev, CL_DRIVER_VERSION, devInfoSize,
        devInfoStr, NULL);
    cl_errChk(status, "Getting vendor name", true);

    return devInfoStr;
#else
#endif
    return nullptr;
}

//! The the name of the device as supplied by the OpenCL implementation
char* cl_getDeviceName(cl_device_id dev)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;
    size_t devInfoSize;
    char* devInfoStr = NULL;

    // If dev is NULL, set it to the default device
    if(dev == NULL) {
        dev = device;
    }

    // Print the name
    status = clGetDeviceInfo(dev, CL_DEVICE_NAME, 0,
        NULL, &devInfoSize);
    cl_errChk(status, "Getting device name", true);

    devInfoStr = (char*)alloc(devInfoSize);

    status = clGetDeviceInfo(dev, CL_DEVICE_NAME, devInfoSize,
        devInfoStr, NULL);
    cl_errChk(status, "Getting device name", true);

    return(devInfoStr);
#else
#endif
    return nullptr;
}

//! Get the name of the vendor for a device
char* cl_getDeviceVendor(cl_device_id dev)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;
    size_t devInfoSize;
    char* devInfoStr = NULL;

    // If dev is NULL, set it to the default device
    if(dev == NULL) {
        dev = device;
    }

    // Print the vendor
    status = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, 0,
        NULL, &devInfoSize);
    cl_errChk(status, "Getting vendor name", true);

    devInfoStr = (char*)alloc(devInfoSize);

    status = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, devInfoSize,
        devInfoStr, NULL);
    cl_errChk(status, "Getting vendor name", true);

    return devInfoStr;
#else
#endif
    return nullptr;
}

//! Get the name of the vendor for a device
char* cl_getDeviceVersion(cl_device_id dev)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;
    size_t devInfoSize;
    char* devInfoStr = NULL;

    // If dev is NULL, set it to the default device
    if(dev == NULL) {
        dev = device;
    }

    // Print the vendor
    status = clGetDeviceInfo(dev, CL_DEVICE_VERSION, 0,
        NULL, &devInfoSize);
    cl_errChk(status, "Getting vendor name", true);

    devInfoStr = (char*)alloc(devInfoSize);

    status = clGetDeviceInfo(dev, CL_DEVICE_VERSION, devInfoSize,
        devInfoStr, NULL);
    cl_errChk(status, "Getting vendor name", true);

    return devInfoStr;
#else
#endif
    return nullptr;
}

//! The the name of the device as supplied by the OpenCL implementation
char* cl_getPlatformName(cl_platform_id platform)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;
    size_t platformInfoSize;
    char* platformInfoStr = NULL;

    // Print the name
    status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0,
        NULL, &platformInfoSize);
    cl_errChk(status, "Getting platform name", true);

    platformInfoStr = (char*)alloc(platformInfoSize);

    status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformInfoSize,
        platformInfoStr, NULL);
    cl_errChk(status, "Getting platform name", true);

    return(platformInfoStr);
#else
#endif
    return nullptr;
}

//! The the name of the device as supplied by the OpenCL implementation
char* cl_getPlatformVendor(cl_platform_id platform)
{
#ifdef  _ORG_CL_DRIVER
    cl_int status;
    size_t platformInfoSize;
    char* platformInfoStr = NULL;

    // Print the name
    status = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0,
        NULL, &platformInfoSize);
    cl_errChk(status, "Getting platform name", true);

    platformInfoStr = (char*)alloc(platformInfoSize);

    status = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformInfoSize,
        platformInfoStr, NULL);
    cl_errChk(status, "Getting platform name", true);

    return(platformInfoStr);
#else
#endif
    return nullptr;
}

//-------------------------------------------------------
//          Utility functions
//-------------------------------------------------------

//! Take a string and an int, and return a string
char* catStringWithInt(const char* string, int integer) {

    if(integer > 99999) {
        printf("Can't handle event identifiers with 6 digits\n");
        exit(-1);
    }

    // 5 characters for the identifier, 1 for the null terminator
    int strLen = strlen(string)+5+1;
    char* eventStr = (char*)alloc(sizeof(char)*strLen);

    char tmp[6];

    strcpy(eventStr, string);
    strcat(eventStr, ",");
    strncat(eventStr, itoa_portable(integer, tmp, 10), 5);

    return eventStr;
}

/**
 ** C++ version 0.4 char* style "itoa":
 ** Written by Lukás Chmela
 ** Released under GPLv3.
 **/
//portable itoa function
char* itoa_portable(int value, char* result, int base) {
    // check that the base if valid
    if (base < 2 || base > 36) { *result = '\0'; return result; }

    char* ptr = result, *ptr1 = result, tmp_char;
    int tmp_value;

    do {
        tmp_value = value;
        value /= base;
        *ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
    } while ( value );

    //Apply negative sign
    if (tmp_value < 0) *ptr++ = '-';
    *ptr-- = '\0';

    while(ptr1 < ptr) {
        tmp_char = *ptr;
        *ptr--= *ptr1;
        *ptr1++ = tmp_char;
    }

    return result;
}

