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

#ifndef __CL_UTILS_H__
#define __CL_UTILS_H__


#define _ORG_CL_DRIVER

#ifdef _ORG_CL_DRIVER
#include <CL/cl.h>
#else
#include "clDriver.h"
#endif 

#include <string>
#include <array>
// The cl_time type is OS specific
#ifdef _WIN32
#include <tchar.h>
#include <Windows.h>
typedef __int64 cl_time; 
#else
#include <sys/time.h>
typedef double cl_time;
#endif


#ifndef _ORG_CL_DRIVER
void dump_tasks();

int coopcl_execute_scan(cl_kernel kernel, cl_uint work_dim, const size_t* global_work_size, const size_t* local_work_size, 
    cl_mem m1, cl_mem m2, int v1, int v2);

int coopcl_execute_transpose(cl_kernel kernel, cl_uint work_dim, const size_t* global_work_size, const size_t* local_work_size, 
    cl_mem m1, cl_mem m2, int v1, int v2);

int coopcl_execute_HessianDet(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size,
    cl_mem arg0, int arg1, int arg2, cl_mem arg3, cl_mem arg4, int arg5, int arg6, int arg7, int arg8);

int coopcl_execute_non_max_supression(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size,
    cl_mem arg0, int arg1, int arg2, int arg3, int arg4, cl_mem arg5, cl_mem arg6,
    int arg7, int arg8, int arg9, cl_mem arg10, int arg11, int arg12, int arg13,
    cl_mem arg14, cl_mem arg15, cl_mem arg16, cl_mem arg17, int arg18, float arg19);

int coopcl_execute_orientation1(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size,
    cl_mem arg0, cl_mem arg1, cl_mem arg2, cl_mem arg3,
    cl_mem arg4, int arg5, int arg6, cl_mem arg7);

int coopcl_execute_orientation2(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size,
    cl_mem arg0, cl_mem arg1);

int coopcl_execute_descriptor(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size,
    cl_mem arg0, int arg1, int arg2, cl_mem arg3, cl_mem arg4,
    cl_mem arg5, cl_mem arg6, cl_mem arg7, cl_mem arg8, cl_mem arg9);

int coopcl_execute_descriptor_norm(cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_size, const size_t* local_work_size,
    cl_mem arg0, cl_mem arg1);

#endif 

//-------------------------------------------------------
// Initialization and Cleanup
//-------------------------------------------------------

// Detects platforms and devices, creates context and command queue
cl_context cl_init(char devicePreference='\0',const float offload=0.0f);

// Releases resources used by clutils
void    cl_cleanup();

// Releases a kernel object
void    cl_freeKernel(cl_kernel kernel);

// Releases a memory object
void    cl_freeMem(cl_mem mem);

// Releases a program object
void    cl_freeProgram(cl_program program);


//-------------------------------------------------------
// Synchronization functions
//-------------------------------------------------------

// Performs a clFinish on the command queue
void    cl_sync();


//-------------------------------------------------------
// Memory allocation
//-------------------------------------------------------

// Allocates a regular buffer on the device
cl_mem  cl_allocBuffer(size_t mem_size, 
            cl_mem_flags flags = CL_MEM_READ_WRITE);

// XXX I don't think this does exactly what we want it to do
// Allocates a read-only buffer and transfers the data
cl_mem  cl_allocBufferConst(size_t mem_size, void* host_ptr);

// Allocates pinned memory on the host
cl_mem  cl_allocBufferPinned(size_t mem_size);

// Allocates an image on the device
cl_mem  cl_allocImage(size_t height, size_t width, char type, 
            cl_mem_flags flags = CL_MEM_READ_WRITE);



//-------------------------------------------------------
// Data transfers
//-------------------------------------------------------

// Copies a buffer from the device to pinned memory on the host and 
// maps it so it can be read
void*   cl_copyAndMapBuffer(cl_mem dst, cl_mem src, size_t size); 

// Copies from one buffer to another
void    cl_copyBufferToBuffer(cl_mem dst, cl_mem src, size_t size);

// Copies data to a buffer on the device
void    cl_copyBufferToDevice(cl_mem dst, void *src, size_t mem_size, 
            cl_bool blocking = CL_TRUE);

// Copies data to an image on the device
void cl_copyImageToDevice(cl_mem dst, void* src, size_t height, size_t width);

// Copies an image from the device to the host
void    cl_copyImageToHost(void* dst, cl_mem src, size_t height, size_t width);

// Copies data from a device buffer to the host
void    cl_copyBufferToHost(void *dst, cl_mem src, size_t mem_size, 
            cl_bool blocking = CL_TRUE);

// Copies data from a buffer on the device to an image on the device
void    cl_copyBufferToImage(cl_mem src, cl_mem dst, int height, int width);

// Maps a buffer
void*   cl_mapBuffer(cl_mem mem, size_t mem_size, cl_mem_flags flags);

// Unmaps a buffer
void    cl_unmapBuffer(cl_mem mem, void *ptr);

// Writes data to a zero-copy buffer on the device
void    cl_writeToZCBuffer(cl_mem mem, void* data, size_t size);

//-------------------------------------------------------
// Program and kernels
//-------------------------------------------------------

// Compiles a program
cl_program  cl_compileProgram(char* kernelPath, char* compileoptions, 
                bool verboseoptions = 0);

// Creates a kernel
cl_kernel   cl_createKernel(cl_program program, const char* kernelName);

// Executes a kernel 
void        cl_executeKernel(cl_kernel kernel, cl_uint work_dim, const size_t* 
                global_work_size, const size_t* local_work_size, 
                const char* description, int identifier = 0);

// Precompiles the kernels for SURF
cl_kernel*  cl_precompileKernels(char* buildOptions,const std::string path="", const std::array<size_t, 3>* global_size=nullptr);

// Sets a kernel argument
void        cl_setKernelArg(cl_kernel kernel, unsigned int index, size_t size, 
                void* data);


//-------------------------------------------------------
// Profiling/events
//-------------------------------------------------------

// Computes the execution time (start to end) for an event
double  cl_computeExecTime(cl_event);

// Compute the elapsed time between two CPU timer values
double  cl_computeTime(cl_time start, cl_time end); 

// Creates an event from CPU timers
void    cl_createUserEvent(cl_time start, cl_time end, char* desc);

// Disable logging of events
void    cl_disableEvents();

// Enable logging of events
void    cl_enableEvents();

// Query the current system time
void    cl_getTime(cl_time* time); 

// Calls a function which prints events to the terminal
void    cl_printEvents();

// Calls a function which writes the events to a file
void    cl_writeEventsToFile(char* path);


//-------------------------------------------------------
// Error handling
//-------------------------------------------------------

// Compare a status value to CL_SUCCESS and optionally exit on error
int     cl_errChk(const cl_int status, const char *msg, bool exitOnErr);

// Queries the supported image formats for the device and prints
// them to the screen
void    printSupportedImageFormats();

//-------------------------------------------------------
// Platform and device information
//-------------------------------------------------------

bool    cl_deviceIsAMD(cl_device_id dev=NULL);
bool    cl_deviceIsNVIDIA(cl_device_id dev=NULL);
bool    cl_platformIsNVIDIA(cl_platform_id plat=NULL);
char*   cl_getDeviceDriverVersion(cl_device_id dev=NULL);
char*   cl_getDeviceName(cl_device_id dev=NULL);
char*   cl_getDeviceVendor(cl_device_id dev=NULL);
char*   cl_getDeviceVersion(cl_device_id dev=NULL);
char*   cl_getPlatformName(cl_platform_id platform);
char*   cl_getPlatformVendor(cl_platform_id platform);

//-------------------------------------------------------
// Utility functions
//-------------------------------------------------------

char* catStringWithInt(const char* str, int integer);

char* itoa_portable(int value, char* result, int base);

//-------------------------------------------------------
// Data types
//-------------------------------------------------------
typedef struct{
    int x;
    int y;
} int2;

typedef struct{
    float x;
    float y;
}float2;

typedef struct{
    float x;
    float y;
    float z;
    float w;
}float4;

//-------------------------------------------------------
// Defines
//-------------------------------------------------------

#define MAX_ERR_VAL 64

#define NUM_PROGRAMS 7
#define NUM_KERNELS 12

#define KERNEL_BUILD_DET 0
#define KERNEL_SURF_DESC 1
#define KERNEL_NORM_DESC 2
#define KERNEL_NON_MAX_SUP 3
#define KERNEL_GET_ORIENT1 4
#define KERNEL_GET_ORIENT2 5
#define KERNEL_NN 6
#define KERNEL_SCAN 7
#define KERNEL_SCAN4 8
#define KERNEL_TRANSPOSE 9
#define KERNEL_SCANIMAGE 10
#define KERNEL_TRANSPOSEIMAGE 11

#endif
