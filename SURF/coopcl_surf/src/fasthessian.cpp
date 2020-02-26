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

#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <vector>

#include "fasthessian.h"
#include "cvutils.h"
#include "clutils.h"
#include "utils.h"


// Based on the octave (row) and interval (column), this lookup table
// identifies the appropriate determinant layer
const int filter_map[MAX_OCTAVES][MAX_INTERVALS] = {{0, 1, 2, 3},
													{1, 3, 4, 5},
													{3, 5, 6, 7},
													{5, 7, 8, 9},
													{7, 9,10,11}};

//-------------------------------------------------------

//! Constructor
FastHessian::FastHessian(int i_height, int i_width, const int octaves,
                         const int intervals, const int sample_step,
                         const float thres, cl_kernel* kernel_list)
                         :kernel_list(kernel_list)
{
    // Initialise variables with bounds-checked values
    this->octaves = (octaves > 0 && octaves <= 4 ? octaves : MAX_OCTAVES);
    this->intervals = (intervals > 0 && intervals <= 4 ? intervals : INTERVALS);
    this->sample_step = (sample_step > 0 && sample_step <= 6 ? sample_step : SAMPLE_STEP);
    this->thres = (thres >= 0 ? thres : THRES);

    this->num_ipts = 0;

    // TODO implement this as device zero-copy memory
    this->d_ipt_count = cl_allocBuffer(sizeof(int));
    cl_copyBufferToDevice(this->d_ipt_count, &this->num_ipts, sizeof(int));
    
    //only for dbg
    //int check = -1;
    //cl_copyBufferToHost(&check, this->d_ipt_count, sizeof(int));

    // Create the hessian response map objects
    this->createResponseMap(octaves, i_width, i_height, sample_step);
}


//! Destructor
FastHessian::~FastHessian()
{
    cl_freeMem(this->d_ipt_count);

    for(unsigned int i = 0; i < this->responseMap.size(); i++) {
        delete responseMap.at(i);
    }
}

void FastHessian::createResponseMap(int octaves, int imgWidth, int imgHeight, int sample_step)
{

    int w = (imgWidth / sample_step);
    int h = (imgHeight / sample_step);
    int s = (sample_step);

    // Calculate approximated determinant of hessian values
    if (octaves >= 1)
    {
        this->responseMap.push_back(new ResponseLayer(w,   h,   s,   9));
        this->responseMap.push_back(new ResponseLayer(w,   h,   s,   15));
        this->responseMap.push_back(new ResponseLayer(w,   h,   s,   21));
        this->responseMap.push_back(new ResponseLayer(w,   h,   s,   27));
    }

    if (octaves >= 2)
    {
        this->responseMap.push_back(new ResponseLayer(w/2, h/2, s*2, 39));
        this->responseMap.push_back(new ResponseLayer(w/2, h/2, s*2, 51));
    }

    if (octaves >= 3)
    {
        this->responseMap.push_back(new ResponseLayer(w/4, h/4, s*4, 75));
        this->responseMap.push_back(new ResponseLayer(w/4, h/4, s*4, 99));
    }

    if (octaves >= 4)
    {
        this->responseMap.push_back(new ResponseLayer(w/8, h/8, s*8, 147));
        this->responseMap.push_back(new ResponseLayer(w/8, h/8, s*8, 195));
    }

    if (octaves >= 5)
    {
        this->responseMap.push_back(new ResponseLayer(w/16, h/16, s*16, 291));
        this->responseMap.push_back(new ResponseLayer(w/16, h/16, s*16, 387));
    }
}


//! Hessian determinant for the image using approximated box filters
/*!
    \param d_intImage Integral Image
    \param surfipt Pointer to pre-allocated temp data structures
    \param i_width Image Width
    \param i_height Image Height
    \param octaves Octaves for SURF
    \param intervals Number of Intervals
    \param kernel_list pointer to precompiled kernels
*/
void FastHessian::computeHessianDet(cl_mem d_intImage,
                                    int i_width, int i_height,
                                    cl_kernel* kernel_list)
{
    this->num_ipts = 0;
    cl_copyBufferToDevice(this->d_ipt_count, &this->num_ipts, sizeof(int));

    // set matrix size and x,y threads per block
    const int BLOCK_DIM = 16;

    cl_kernel hessian_det =  kernel_list[KERNEL_BUILD_DET];

    size_t localWorkSize[2] = {BLOCK_DIM,BLOCK_DIM};
    size_t globalWorkSize[2];
	

    for(unsigned int i = 0; i < this->responseMap.size(); i++) 
    {
        cl_mem responses = this->responseMap.at(i)->getResponses();
        cl_mem laplacian = this->responseMap.at(i)->getLaplacian();
        int step = this->responseMap.at(i)->getStep();
        int filter = this->responseMap.at(i)->getFilter();
        int layerWidth = this->responseMap.at(i)->getWidth();
        int layerHeight = this->responseMap.at(i)->getHeight();

        globalWorkSize[0] = roundUp(layerWidth, localWorkSize[0]);
        globalWorkSize[1] = roundUp(layerHeight, localWorkSize[1]);

#ifdef _ORG_CL_DRIVER
        cl_setKernelArg(hessian_det, 0, sizeof(cl_mem), (void*)&d_intImage);
        cl_setKernelArg(hessian_det, 1, sizeof(cl_int), (void*)&i_width);
        cl_setKernelArg(hessian_det, 2, sizeof(cl_int), (void*)&i_height);
        cl_setKernelArg(hessian_det, 3, sizeof(cl_mem), (void*)&responses);
        cl_setKernelArg(hessian_det, 4, sizeof(cl_mem), (void*)&laplacian);
        cl_setKernelArg(hessian_det, 5, sizeof(int),    (void*)&layerWidth);
        cl_setKernelArg(hessian_det, 6, sizeof(int),    (void*)&layerHeight);
        cl_setKernelArg(hessian_det, 7, sizeof(int),    (void*)&step);
        cl_setKernelArg(hessian_det, 8, sizeof(int),    (void*)&filter);
        cl_executeKernel(hessian_det, 2, globalWorkSize, localWorkSize,"BuildHessianDet", i);
#else
        coopcl_execute_HessianDet(hessian_det, 2, globalWorkSize, localWorkSize,
            d_intImage, i_width, i_height, responses, laplacian, layerWidth, layerHeight, step, filter);
        
#endif
        // TODO Verify that a clFinish is not required (setting an argument
        //      to the loop counter without it may be problematic, but it
        //      really kills performance on AMD parts)
        //cl_sync();
    }
}


/*!
    Find the image features and write into vector of features
    Determine what points are interesting and store them
    \param img
    \param d_intImage The integral image pointer on the device
    \param d_laplacian
    \param d_pixPos
    \param d_scale
*/
int FastHessian::getIpoints(cv::Mat& img, cl_mem d_intImage, cl_mem d_laplacian,
                            cl_mem d_pixPos, cl_mem d_scale, int maxIpts)
{
	// Compute the hessian determinants
    // GPU kernels: init_det and build_det kernels
    this->computeHessianDet(d_intImage, img.cols, img.rows, kernel_list);

	// Determine which points are interesting
    // GPU kernels: non_max_suppression kernel
    this->selectIpoints(d_laplacian, d_pixPos, d_scale, kernel_list, maxIpts);

	// Copy the number of interesting points back to the host
    cl_copyBufferToHost(&this->num_ipts, this->d_ipt_count, sizeof(int));

	// Sanity check
    if(this->num_ipts < 0) {
        printf("Invalid number of Ipoints\n");
        exit(-1);
    };

    return num_ipts;
}

/*!
//! Calculate the position of ipoints (gpuIpoint::d_pixPos) using non maximal suppression

    Convert d_m_det which is a array of all the hessians into d_pixPos
    which is a float2 array of the (x,y) of all ipoint locations
    \param i_width The width of the image
    \param i_height The height of the image
    \param d_laplacian
    \param d_pixPos
    \param d_scale
    \param kernel_list Precompiled Kernels
*/
void FastHessian::selectIpoints(cl_mem d_laplacian, cl_mem d_pixPos,
                                cl_mem d_scale, cl_kernel* kernel_list,
                                int maxPoints)
{

    // The search for exterema (the most interesting point in a neighborhood)
    // is done by non-maximal suppression

    cl_kernel non_max_supression = kernel_list[KERNEL_NON_MAX_SUP];

    int BLOCK_W=16;
    int BLOCK_H=16;
    
    // Run the kernel for each octave
    for(int o = 0; o < octaves; o++)
    {
        for(int i = 0; i <= 1; i++) 
        {
            cl_mem bResponse = this->responseMap.at(filter_map[o][i])->getResponses();
            int bWidth = this->responseMap.at(filter_map[o][i])->getWidth();
            int bHeight = this->responseMap.at(filter_map[o][i])->getHeight();
            int bFilter = this->responseMap.at(filter_map[o][i])->getFilter();

            cl_mem mResponse = this->responseMap.at(filter_map[o][i+1])->getResponses();
            int mWidth = this->responseMap.at(filter_map[o][i+1])->getWidth();
            int mHeight = this->responseMap.at(filter_map[o][i+1])->getHeight();
            int mFilter = this->responseMap.at(filter_map[o][i+1])->getFilter();
            cl_mem mLaplacian = this->responseMap.at(filter_map[o][i+1])->getLaplacian();

            cl_mem tResponse = this->responseMap.at(filter_map[o][i+2])->getResponses();
            int tWidth = this->responseMap.at(filter_map[o][i+2])->getWidth();
            int tHeight = this->responseMap.at(filter_map[o][i+2])->getHeight();
            int tFilter = this->responseMap.at(filter_map[o][i+2])->getFilter();
            int tStep = this->responseMap.at(filter_map[o][i+2])->getStep();

            size_t localWorkSize[2] = {BLOCK_W, BLOCK_H};
            size_t globalWorkSize[2] = {roundUp(mWidth, BLOCK_W),roundUp(mHeight, BLOCK_H)};
#ifdef _ORG_CL_DRIVER
            cl_setKernelArg(non_max_supression,  0, sizeof(cl_mem), (void*)&tResponse);
            cl_setKernelArg(non_max_supression,  1, sizeof(int),    (void*)&tWidth);
            cl_setKernelArg(non_max_supression,  2, sizeof(int),    (void*)&tHeight);
            cl_setKernelArg(non_max_supression,  3, sizeof(int),    (void*)&tFilter);
            cl_setKernelArg(non_max_supression,  4, sizeof(int),    (void*)&tStep);
            cl_setKernelArg(non_max_supression,  5, sizeof(cl_mem), (void*)&mResponse);
            cl_setKernelArg(non_max_supression,  6, sizeof(cl_mem), (void*)&mLaplacian);
            cl_setKernelArg(non_max_supression,  7, sizeof(int),    (void*)&mWidth);
            cl_setKernelArg(non_max_supression,  8, sizeof(int),    (void*)&mHeight);
            cl_setKernelArg(non_max_supression,  9, sizeof(int),    (void*)&mFilter);
            cl_setKernelArg(non_max_supression, 10, sizeof(cl_mem), (void*)&bResponse);
            cl_setKernelArg(non_max_supression, 11, sizeof(int),    (void*)&bWidth);
            cl_setKernelArg(non_max_supression, 12, sizeof(int),    (void*)&bHeight);
            cl_setKernelArg(non_max_supression, 13, sizeof(int),    (void*)&bFilter);
            cl_setKernelArg(non_max_supression, 14, sizeof(cl_mem), (void*)&(this->d_ipt_count));
            cl_setKernelArg(non_max_supression, 15, sizeof(cl_mem), (void*)&d_pixPos);
            cl_setKernelArg(non_max_supression, 16, sizeof(cl_mem), (void*)&d_scale);
            cl_setKernelArg(non_max_supression, 17, sizeof(cl_mem), (void*)&d_laplacian);
            cl_setKernelArg(non_max_supression, 18, sizeof(int), (void*)&maxPoints);
            cl_setKernelArg(non_max_supression, 19, sizeof(float), (void*)&(this->thres));
            // Call non-max supression kernel
            cl_executeKernel(non_max_supression, 2, globalWorkSize, localWorkSize,"NonMaxSupression", o*2+i);
#else            
            coopcl_execute_non_max_supression(non_max_supression, 2, globalWorkSize, localWorkSize, 
                tResponse,tWidth,tHeight,tFilter,tStep,mResponse,mLaplacian,mWidth,mHeight,mFilter,bResponse,
                bWidth,bHeight,bFilter,d_ipt_count,d_pixPos,d_scale,d_laplacian,maxPoints,thres);

#endif
            // TODO Verify that a clFinish is not required (setting an argument
            //      to the loop counter without it may be problematic, but it
            //      really kills performance on AMD parts)
            //cl_sync();
        }
    }
}


//! Reset the state of the data
void FastHessian::reset()
{
    int numIpts = 0;
    cl_copyBufferToDevice(this->d_ipt_count, &numIpts, sizeof(int));
}