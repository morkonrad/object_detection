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
#include <time.h>

#include "cvutils.h"
#include "clutils.h"
#include "utils.h"

#include "surf.h"
#include "eventlist.h"
#include "stdio.h"


// TODO Get rid of these arrays (i and j).  Have the values computed 
//      dynamically within the kernel
const int Surf::j[] = {-12, -7, -2, 3,
                       -12, -7, -2, 3,
                       -12, -7, -2, 3,
                       -12, -7, -2, 3};

const int Surf::i[] = {-12,-12,-12,-12,
                        -7, -7, -7, -7,
                        -2, -2, -2, -2,
                         3,  3,  3,  3};

const unsigned int Surf::id[] = {6,5,4,3,2,1,0,1,2,3,4,5,6};

const float Surf::gauss25[] = {
    0.02350693969273f, 0.01849121369071f, 0.01239503121241f, 0.00708015417522f, 0.00344628101733f, 0.00142945847484f, 0.00050524879060f,
    0.02169964028389f, 0.01706954162243f, 0.01144205592615f, 0.00653580605408f, 0.00318131834134f, 0.00131955648461f, 0.00046640341759f,
    0.01706954162243f, 0.01342737701584f, 0.00900063997939f, 0.00514124713667f, 0.00250251364222f, 0.00103799989504f, 0.00036688592278f,
    0.01144205592615f, 0.00900063997939f, 0.00603330940534f, 0.00344628101733f, 0.00167748505986f, 0.00069579213743f, 0.00024593098864f,
    0.00653580605408f, 0.00514124713667f, 0.00344628101733f, 0.00196854695367f, 0.00095819467066f, 0.00039744277546f, 0.00014047800980f,
    0.00318131834134f, 0.00250251364222f, 0.00167748505986f, 0.00095819467066f, 0.00046640341759f, 0.00019345616757f, 0.00006837798818f,
    0.00131955648461f, 0.00103799989504f, 0.00069579213743f, 0.00039744277546f, 0.00019345616757f, 0.00008024231247f, 0.00002836202103f};

#ifdef _USE_OCL_

//! Constructor
Surf::Surf(int initialPoints, int i_height, int i_width, int octaves, 
           int intervals, int sample_step, float threshold,
           cl_kernel* kernel_list)
           : kernel_list(kernel_list)
{
	//device = std::make_unique<coopcl::virtual_device>();

    this->fh = new FastHessian(i_height, i_width, 
		octaves, intervals, sample_step, threshold, kernel_list);

    // Once we know the size of the image, successive frames should stay
    // the same size, so we can just allocate the space once for the integral
    // image and intermediate data
    if(isUsingImages()) 
    {   
        this->d_intImage = cl_allocImage(i_height, i_width, 'f');
        this->d_tmpIntImage = cl_allocImage(i_height, i_width, 'f');
        this->d_tmpIntImageT1 = cl_allocImage(i_width, i_height, 'f');
        this->d_tmpIntImageT2 = cl_allocImage(i_width, i_height, 'f');
    }
    else {
        this->d_intImage = cl_allocBuffer(sizeof(float)*i_width*i_height);
        this->d_tmpIntImage = cl_allocBuffer(sizeof(float)*i_height*i_width);
        // These two are unnecessary for buffers, but required for images, so
        // we'll use them for buffers as well to keep the code clean
        this->d_tmpIntImageT1 = cl_allocBuffer(sizeof(float)*i_height*i_width);
        this->d_tmpIntImageT2 = cl_allocBuffer(sizeof(float)*i_height*i_width);
    }

    // Allocate constant data on device
    this->d_gauss25 = cl_allocBufferConst(sizeof(float)*49,(void*)Surf::gauss25);
    this->d_id = cl_allocBufferConst(sizeof(unsigned int)*13,(void*)Surf::id);
    this->d_i = cl_allocBufferConst(sizeof(int)*16,(void*)Surf::i);
    this->d_j = cl_allocBufferConst(sizeof(int)*16,(void*)Surf::j);

    // Allocate buffers for each of the interesting points.  We don't know
    // how many there are initially, so must allocate more than enough space
    this->d_scale = cl_allocBuffer(initialPoints * sizeof(float));
    this->d_pixPos = cl_allocBuffer(initialPoints * sizeof(float2));
    this->d_laplacian = cl_allocBuffer(initialPoints * sizeof(int));
    
    // These buffers used to wait for the number of actual ipts to be known
    // before being allocated, instead now we'll only allocate them once
    // so that we can take advantage of optimized data transfers and reallocate
    // them if there's not enough space available
    this->d_length = cl_allocBuffer(initialPoints * DESC_SIZE * sizeof(float));
    this->d_desc = cl_allocBuffer(initialPoints * DESC_SIZE * sizeof(float));
    this->d_res = cl_allocBuffer(initialPoints * 109 * sizeof(float4));
    this->d_orientation = cl_allocBuffer(initialPoints * sizeof(float));

    // Allocate buffers to store the output data (descriptor information)
    // on the host
#ifdef OPTIMIZED_TRANSFERS
    this->h_scale = cl_allocBufferPinned(initialPoints * sizeof(float));
    this->h_pixPos = cl_allocBufferPinned(initialPoints * sizeof(float2));
    this->h_laplacian = cl_allocBufferPinned(initialPoints * sizeof(int));
    this->h_desc = cl_allocBufferPinned(initialPoints * DESC_SIZE * sizeof(float));
    this->h_orientation = cl_allocBufferPinned(initialPoints * sizeof(float));
#else
    this->scale = (float*)alloc(initialPoints * sizeof(float));
    this->pixPos = (float2*)alloc(initialPoints * sizeof(float2));
    this->laplacian = (int*)alloc(initialPoints * sizeof(int));
    this->desc = (float*)alloc(initialPoints * DESC_SIZE * sizeof(float));
    this->orientation = (float*)alloc(initialPoints * sizeof(float));
#endif
    // This is how much space is available for Ipts
    this->maxIpts = initialPoints;
}


//! Destructor
Surf::~Surf() {

    cl_freeMem(this->d_intImage);
    cl_freeMem(this->d_tmpIntImage);
    cl_freeMem(this->d_tmpIntImageT1);
    cl_freeMem(this->d_tmpIntImageT2);
    cl_freeMem(this->d_desc);
    cl_freeMem(this->d_orientation);
    cl_freeMem(this->d_gauss25);
    cl_freeMem(this->d_id);
    cl_freeMem(this->d_i);
    cl_freeMem(this->d_j);
    cl_freeMem(this->d_laplacian);
    cl_freeMem(this->d_pixPos);
    cl_freeMem(this->d_scale);
    cl_freeMem(this->d_res);
    cl_freeMem(this->d_length);

#ifdef OPTIMIZED_TRANSFERS
    cl_freeMem(this->h_orientation);
    cl_freeMem(this->h_scale);
    cl_freeMem(this->h_laplacian);
    cl_freeMem(this->h_desc);
    cl_freeMem(this->h_pixPos);
#else
    free(this->orientation);
    free(this->scale);
    free(this->laplacian);
    free(this->desc);
    free(this->pixPos);
#endif

    delete this->fh;
}

//! Computes the integral image of image img.
//! Assumes source image to be a  32-bit floating point.
/*!
    Saves integral Image in d_intImage on the GPU
    \param source Input Image as grabbed by OpenCv
*/
void Surf::computeIntegralImage(cv::Mat&  source)
{
    //! convert the image to single channel 32f

    // TODO This call takes about 4ms (is there any way to speed it up?)
    cv::Mat img = getGray(source);

    // set up variables for data access
    int height = img.rows;
    int width = img.cols;
    float *data = (float*)img.data;

    cl_kernel scan_kernel;
    cl_kernel transpose_kernel;

    if(isUsingImages()) {
        // Copy the data to the GPU
        cl_copyImageToDevice(this->d_intImage, data, height, width);

        scan_kernel = this->kernel_list[KERNEL_SCANIMAGE];
        transpose_kernel = this->kernel_list[KERNEL_TRANSPOSEIMAGE];
    }
    else {
        // Copy the data to the GPU
        cl_copyBufferToDevice(this->d_intImage, data, sizeof(float)*width*height);

        // If it is possible to use the vector scan (scan4) use
        // it, otherwise, use the regular scan
        if(cl_deviceIsAMD() && width % 4 == 0 && height % 4 == 0) 
        {
            // NOTE Change this to KERNEL_SCAN when running verification code.
            //      The reference code doesn't use a vector type and
            //      scan4 produces a slightly different integral image
            scan_kernel = this->kernel_list[KERNEL_SCAN4];
        }
        else 
        {
            scan_kernel = this->kernel_list[KERNEL_SCAN];
        }
        transpose_kernel = this->kernel_list[KERNEL_TRANSPOSE];
    }
    
    // -----------------------------------------------------------------
    // Step 1: Perform integral summation on the rows
    // -----------------------------------------------------------------
    size_t localWorkSize1[2]={64, 1};
    size_t globalWorkSize1[2]={64, (size_t)height};
#ifdef _ORG_CL_DRIVER
    cl_setKernelArg(scan_kernel, 0, sizeof(cl_mem), (void *)&(this->d_intImage));
    cl_setKernelArg(scan_kernel, 1, sizeof(cl_mem), (void *)&(this->d_tmpIntImage)); 
    cl_setKernelArg(scan_kernel, 2, sizeof(int), (void *)&height);
    cl_setKernelArg(scan_kernel, 3, sizeof(int), (void *)&width);
    cl_executeKernel(scan_kernel, 2, globalWorkSize1, localWorkSize1, "Scan", 0);
#else
    coopcl_execute_scan(scan_kernel, 2, globalWorkSize1, localWorkSize1, this->d_intImage, this->d_tmpIntImage, height, width);
#endif 
    // -----------------------------------------------------------------
    // Step 2: Transpose
    // -----------------------------------------------------------------

    size_t localWorkSize2[]={16, 16};
    size_t globalWorkSize2[]={roundUp(width,16), roundUp(height,16)};
#ifdef _ORG_CL_DRIVER
    cl_setKernelArg(transpose_kernel, 0, sizeof(cl_mem), (void *)&(this->d_tmpIntImage));  
    cl_setKernelArg(transpose_kernel, 1, sizeof(cl_mem), (void *)&(this->d_tmpIntImageT1)); 
    cl_setKernelArg(transpose_kernel, 2, sizeof(int), (void *)&height);
    cl_setKernelArg(transpose_kernel, 3, sizeof(int), (void *)&width);
    cl_executeKernel(transpose_kernel, 2, globalWorkSize2, localWorkSize2, "Transpose", 0);
#else
    coopcl_execute_transpose(transpose_kernel, 2, globalWorkSize2, localWorkSize2, this->d_tmpIntImage, this->d_tmpIntImageT1, height, width);
#endif

    // -----------------------------------------------------------------
    // Step 3: Run integral summation on the rows again (same as columns
    //         integral since we've transposed). 
    // -----------------------------------------------------------------

    int heightT = width;
    int widthT = height;

    size_t localWorkSize3[2]={64, 1};
    size_t globalWorkSize3[2]={64, (size_t)heightT};
#ifdef _ORG_CL_DRIVER
    cl_setKernelArg(scan_kernel, 0, sizeof(cl_mem), (void *)&(this->d_tmpIntImageT1));
    cl_setKernelArg(scan_kernel, 1, sizeof(cl_mem), (void *)&(this->d_tmpIntImageT2)); 
    cl_setKernelArg(scan_kernel, 2, sizeof(int), (void *)&heightT);
    cl_setKernelArg(scan_kernel, 3, sizeof(int), (void *)&widthT);
    cl_executeKernel(scan_kernel, 2, globalWorkSize3, localWorkSize3, "Scan", 1);
#else
    coopcl_execute_scan(scan_kernel, 2, globalWorkSize3, localWorkSize3, this->d_tmpIntImageT1, this->d_tmpIntImageT2, heightT, widthT);
#endif
    // -----------------------------------------------------------------
    // Step 4: Transpose back
    // -----------------------------------------------------------------

    size_t localWorkSize4[]={16, 16};
    size_t globalWorkSize4[]={roundUp(widthT,16), roundUp(heightT,16)};
#ifdef _ORG_CL_DRIVER
    cl_setKernelArg(transpose_kernel, 0, sizeof(cl_mem), (void *)&(this->d_tmpIntImageT2)); 
    cl_setKernelArg(transpose_kernel, 1, sizeof(cl_mem), (void *)&(this->d_intImage));
    cl_setKernelArg(transpose_kernel, 2, sizeof(int), (void *)&heightT);
    cl_setKernelArg(transpose_kernel, 3, sizeof(int), (void *)&widthT);
    cl_executeKernel(transpose_kernel, 2, globalWorkSize4, localWorkSize4, "Transpose", 1);
#else
    coopcl_execute_transpose(transpose_kernel, 2, globalWorkSize4, localWorkSize4, this->d_tmpIntImageT2, this->d_intImage, heightT, widthT);
#endif

    // release the gray image
    //cvReleaseImage(&img);
}


//! Create the SURF descriptors
/*!
    Calculate orientation for all ipoints using the
    sliding window technique from OpenSurf
    \param d_intImage The integral image
    \param width The width of the image
    \param height The height of the image
*/
void Surf::createDescriptors(int i_width, int i_height)
{
    const size_t threadsPerWG = DESC_TH_WGS;
    const size_t wgsPerIpt = DESC_WGS_ITEM;

    cl_kernel surf64Descriptor_kernel = this->kernel_list[KERNEL_SURF_DESC];
    size_t localWorkSizeSurf64[2] = {threadsPerWG,1};
    size_t globalWorkSizeSurf64[2] = {(wgsPerIpt*threadsPerWG),(size_t)numIpts};

#ifdef _ORG_CL_DRIVER
    cl_setKernelArg(surf64Descriptor_kernel, 0, sizeof(cl_mem), (void*)&(this->d_intImage));
    cl_setKernelArg(surf64Descriptor_kernel, 1, sizeof(int),    (void*)&i_width);
    cl_setKernelArg(surf64Descriptor_kernel, 2, sizeof(int),    (void*)&i_height);
    cl_setKernelArg(surf64Descriptor_kernel, 3, sizeof(cl_mem), (void*)&(this->d_scale));
    cl_setKernelArg(surf64Descriptor_kernel, 4, sizeof(cl_mem), (void*)&(this->d_desc));
    cl_setKernelArg(surf64Descriptor_kernel, 5, sizeof(cl_mem), (void*)&(this->d_pixPos));
    cl_setKernelArg(surf64Descriptor_kernel, 6, sizeof(cl_mem), (void*)&(this->d_orientation));
    cl_setKernelArg(surf64Descriptor_kernel, 7, sizeof(cl_mem), (void*)&(this->d_length));
    cl_setKernelArg(surf64Descriptor_kernel, 8, sizeof(cl_mem), (void*)&(this->d_j));
    cl_setKernelArg(surf64Descriptor_kernel, 9, sizeof(cl_mem), (void*)&(this->d_i));
    // Execute the descriptor kernel
    cl_executeKernel(surf64Descriptor_kernel, 2, globalWorkSizeSurf64,localWorkSizeSurf64, "CreateDescriptors"); 
#else
    coopcl_execute_descriptor(surf64Descriptor_kernel, 2, globalWorkSizeSurf64, localWorkSizeSurf64,
        this->d_intImage, i_width, i_height, d_scale, this->d_desc, d_pixPos, d_orientation,
        d_length, d_j, d_i);
#endif

    cl_kernel normSurf64_kernel = kernel_list[KERNEL_NORM_DESC];
    size_t localWorkSizeNorm64[] = {DESC_SIZE};
    size_t globallWorkSizeNorm64[] =  {(size_t)(this->numIpts*DESC_SIZE)};

#ifdef _ORG_CL_DRIVER
    cl_setKernelArg(normSurf64_kernel, 0, sizeof(cl_mem), (void*)&(this->d_desc));
    cl_setKernelArg(normSurf64_kernel, 1, sizeof(cl_mem), (void*)&(this->d_length));
    // Execute the descriptor normalization kernel
    cl_executeKernel(normSurf64_kernel, 1, globallWorkSizeNorm64, localWorkSizeNorm64,"NormalizeDescriptors"); 
#else
    
    coopcl_execute_descriptor_norm(normSurf64_kernel, 1, globallWorkSizeNorm64, localWorkSizeNorm64, d_desc, d_length);

#endif


} 

//! Calculate orientation for all ipoints
/*!
    Calculate orientation for all ipoints using the
    sliding window technique from OpenSurf
    \param i_width The image width
    \param i_height The image height
*/
void Surf::getOrientations(int i_width, int i_height)
{
    cl_kernel getOrientation = this->kernel_list[KERNEL_GET_ORIENT1];
    cl_kernel getOrientation2 = this->kernel_list[KERNEL_GET_ORIENT2];  

    size_t localWorkSize1[] = {ORIENT_WGS_STEP1};
    size_t globalWorkSize1[] = {(size_t)this->numIpts*(size_t)ORIENT_WGS_STEP1 };

    /*!
    Assign the supplied Ipoint an orientation
    */
#ifdef _ORG_CL_DRIVER
    cl_setKernelArg(getOrientation, 0, sizeof(cl_mem), (void *)&(this->d_intImage));
    cl_setKernelArg(getOrientation, 1, sizeof(cl_mem), (void *)&(this->d_scale));
    cl_setKernelArg(getOrientation, 2, sizeof(cl_mem), (void *)&(this->d_pixPos));
    cl_setKernelArg(getOrientation, 3, sizeof(cl_mem), (void *)&(this->d_gauss25));
    cl_setKernelArg(getOrientation, 4, sizeof(cl_mem), (void *)&(this->d_id));
    cl_setKernelArg(getOrientation, 5, sizeof(int),    (void *)&i_width);
    cl_setKernelArg(getOrientation, 6, sizeof(int),    (void *)&i_height);
    cl_setKernelArg(getOrientation, 7, sizeof(cl_mem), (void *)&(this->d_res));
    // Execute the kernel
    cl_executeKernel(getOrientation, 1, globalWorkSize1, localWorkSize1, "GetOrientations");
#else
    coopcl_execute_orientation1(getOrientation, 1, globalWorkSize1, localWorkSize1,
        this->d_intImage, this->d_scale, this->d_pixPos, this->d_gauss25, this->d_id, i_width, i_height, this->d_res);
#endif

    size_t localWorkSize2[] = { ORIENT_WGS_STEP2 };
    size_t globalWorkSize2[] = { (size_t)numIpts * (size_t)ORIENT_WGS_STEP2 };
#ifdef _ORG_CL_DRIVER
    cl_setKernelArg(getOrientation2, 0, sizeof(cl_mem), (void *)&(this->d_orientation));
    cl_setKernelArg(getOrientation2, 1, sizeof(cl_mem), (void *)&(this->d_res));    
    // Execute the kernel
    cl_executeKernel(getOrientation2, 1, globalWorkSize2, localWorkSize2,"GetOrientations2");
#else
    coopcl_execute_orientation2(getOrientation2, 1, globalWorkSize2, localWorkSize2, d_orientation, d_res);
#endif
}

//! Allocates the memory objects required for the ipt descriptor information
void Surf::reallocateIptBuffers() {

    // Release the old memory objects (that were too small)
    cl_freeMem(d_scale);
    cl_freeMem(d_pixPos);
    cl_freeMem(d_laplacian);
    cl_freeMem(d_length);
    cl_freeMem(d_desc);
    cl_freeMem(d_res);
    cl_freeMem(d_orientation);

    free(this->orientation);
    free(this->scale);
    free(this->laplacian);
    free(this->desc);
    free(this->pixPos);

    int newSize = this->maxIpts;

    // Allocate new memory objects based on the new size
    this->d_scale = cl_allocBuffer(newSize * sizeof(float));
    this->d_pixPos = cl_allocBuffer(newSize * sizeof(float2));
    this->d_laplacian = cl_allocBuffer(newSize * sizeof(int));
    this->d_length = cl_allocBuffer(newSize * DESC_SIZE*sizeof(float));
    this->d_desc = cl_allocBuffer(newSize * DESC_SIZE * sizeof(float));
    this->d_res = cl_allocBuffer(newSize * 121 * sizeof(float4));
    this->d_orientation = cl_allocBuffer(newSize * sizeof(float));

#ifdef OPTIMIZED_TRANSFERS
    this->h_scale = cl_allocBufferPinned(newSize * sizeof(float));
    this->h_pixPos = cl_allocBufferPinned(newSize * sizeof(float2));
    this->h_laplacian = cl_allocBufferPinned(newSize * sizeof(int));
    this->h_desc = cl_allocBufferPinned(newSize * DESC_SIZE * sizeof(float));
    this->h_orientation = cl_allocBufferPinned(newSize * sizeof(float));
#else
    this->scale = (float*)alloc(newSize * sizeof(float));
    this->pixPos = (float2*)alloc(newSize * sizeof(float2));
    this->laplacian = (int*)alloc(newSize * sizeof(int));
    this->desc = (float*)alloc(newSize * DESC_SIZE * sizeof(float));
    this->orientation = (float*)alloc(newSize * sizeof(float));
#endif
}


//! This function gets called each time SURF is run on a new frame.  It prevents
//! having to create and destroy the object each time (lots of OpenCL overhead)
void Surf::reset() 
{
    this->fh->reset();
}


//! Retreive the descriptors from the GPU
/*!
    Copy data back from the GPU into an IpVec structure on the host
*/
IpVec* Surf::retrieveDescriptors()
{
    IpVec* ipts = new IpVec();

    if(this->numIpts == 0) 
    {
        return ipts;
    }

    // Copy back the output data

#ifdef OPTIMIZED_TRANSFERS
    // We're using pinned memory for the transfers.  The data is 
    // copied back to pinned memory and then must be mapped before
    // it's usable on the host

    // Copy back Laplacian data
    this->laplacian = (int*)cl_copyAndMapBuffer(this->h_laplacian, 
        this->d_laplacian, this->numIpts * sizeof(int));

    // Copy back scale data
    this->scale = (float*)cl_copyAndMapBuffer(this->h_scale, 
        this->d_scale, this->numIpts * sizeof(float));
    
    // Copy back pixel positions
    this->pixPos = (float2*)cl_copyAndMapBuffer(this->h_pixPos, 
        this->d_pixPos, this->numIpts * sizeof(float2));

    // Copy back descriptors
    this->desc = (float*)cl_copyAndMapBuffer(this->h_desc, 
        this->d_desc, this->numIpts * DESC_SIZE* sizeof(float));

    // Copy back orientation data
    this->orientation = (float*)cl_copyAndMapBuffer(this->h_orientation, 
        this->d_orientation, this->numIpts * sizeof(float));
#else
    // Copy back Laplacian information
    cl_copyBufferToHost(this->laplacian, this->d_laplacian, 
        (this->numIpts) * sizeof(int), CL_FALSE);

    // Copy back scale data
    cl_copyBufferToHost(this->scale, this->d_scale,
        (this->numIpts)*sizeof(float), CL_FALSE);

    // Copy back pixel positions
    cl_copyBufferToHost(this->pixPos, this->d_pixPos, 
        (this->numIpts) * sizeof(float2), CL_FALSE);   

    // Copy back descriptors
    cl_copyBufferToHost(this->desc, this->d_desc, 
        (this->numIpts)*DESC_SIZE*sizeof(float), CL_FALSE);
    
    // Copy back orientation data
    cl_copyBufferToHost(this->orientation, this->d_orientation, 
        (this->numIpts)*sizeof(float), CL_TRUE);
#endif  

    // Parse the data into Ipoint structures
    for(int i= 0;i<(this->numIpts);i++)
    {		
        Ipoint ipt;		
        ipt.x = pixPos[i].x;
        ipt.y = pixPos[i].y;
        ipt.scale = scale[i];
        ipt.laplacian = laplacian[i];
        ipt.orientation = orientation[i];
        memcpy(ipt.descriptor, &desc[i*64], sizeof(float)*64);
        ipts->push_back(ipt);
    }

#ifdef OPTIMIZED_TRANSFERS
    // We're done reading from the buffers, so we unmap
    // them so they can be used again by the device
    cl_unmapBuffer(this->h_laplacian, this->laplacian);
    cl_unmapBuffer(this->h_scale, this->scale);
    cl_unmapBuffer(this->h_pixPos, this->pixPos);
    cl_unmapBuffer(this->h_desc, this->desc);
    cl_unmapBuffer(this->h_orientation, this->orientation);
#endif

    return ipts;
}


//! Function that builds vector of interest points.  This is the main SURF function
//! that will be called for any type of input.
/*!
    High level driver function for entire OpenSurfOpenCl
    \param img image to find Ipoints within
    \param upright Switch for future functionality of upright surf
    \param fh FastHessian object
*/
void Surf::run(cv::Mat&  img, bool upright) 
{

    if (upright)
    {
        // Extract upright (i.e. not rotation invariant) descriptors
        printf("Upright surf not supported\n");
        exit(1);		
    }  
    
    // Perform the scan sum of the image (populates d_intImage)
    // GPU kernels: scan (x2), tranpose (x2)
    this->computeIntegralImage(img);

    // Determines the points of interest
    // GPU kernels: init_det, hessian_det (x12), non_max_suppression (x3)
    // GPU mem transfer: copies back the number of ipoints 
    
	this->numIpts = this->fh->getIpoints(img, 
		this->d_intImage, this->d_laplacian,
		this->d_pixPos, this->d_scale, this->maxIpts);

    // Verify that there was enough space allocated for the number of
    // Ipoints found
    if(this->numIpts >= this->maxIpts) 
	{
        // If not enough space existed, we need to reallocate space and
        // run the kernels again
        printf("Not enough space for Ipoints, reallocating and running again\n");
        this->maxIpts = this->numIpts * 2;
        
		this->reallocateIptBuffers();        
        this->fh->reset();

        numIpts = fh->getIpoints(img, this->d_intImage, this->d_laplacian, 
		this->d_pixPos, this->d_scale, this->maxIpts);
		/*numIpts = fh->getIpoints(img, *ccl_d_intImage, *ccl_d_laplacian,
			*ccl_d_pixPos, *ccl_d_scale, maxIpts);*/
    }

    printf("There were %d interest points\n", this->numIpts);    

    // Main SURF-64 loop assigns orientations and gets descriptors    
    if(this->numIpts==0) return;

    // GPU kernel: getOrientation1 (1x), getOrientation2 (1x)
    this->getOrientations(img.cols, img.rows);

    // GPU kernel: surf64descriptor (1x), norm64descriptor (1x)
    this->createDescriptors(img.cols, img.rows);
}

#else
//! Constructor
Surf::Surf(int initialPoints, int i_height, int i_width, int octaves,
	int intervals, int sample_step, float threshold,
	cl_kernel* kernel_list)
	: kernel_list(kernel_list)
{

	this->fh = new FastHessian(i_height, i_width, octaves,
		intervals, sample_step, threshold, kernel_list);

	// Once we know the size of the image, successive frames should stay
	// the same size, so we can just allocate the space once for the integral
	// image and intermediate data
	if (isUsingImages())
	{
		this->d_intImage = cl_allocImage(i_height, i_width, 'f');
		this->d_tmpIntImage = cl_allocImage(i_height, i_width, 'f');
		this->d_tmpIntImageT1 = cl_allocImage(i_width, i_height, 'f');
		this->d_tmpIntImageT2 = cl_allocImage(i_width, i_height, 'f');
	}
	else {
		this->d_intImage = cl_allocBuffer(sizeof(float)*i_width*i_height);
		this->d_tmpIntImage = cl_allocBuffer(sizeof(float)*i_height*i_width);
		// These two are unnecessary for buffers, but required for images, so
		// we'll use them for buffers as well to keep the code clean
		this->d_tmpIntImageT1 = cl_allocBuffer(sizeof(float)*i_height*i_width);
		this->d_tmpIntImageT2 = cl_allocBuffer(sizeof(float)*i_height*i_width);
	}

	// Allocate constant data on device
	this->d_gauss25 = cl_allocBufferConst(sizeof(float) * 49, (void*)Surf::gauss25);
	this->d_id = cl_allocBufferConst(sizeof(unsigned int) * 13, (void*)Surf::id);
	this->d_i = cl_allocBufferConst(sizeof(int) * 16, (void*)Surf::i);
	this->d_j = cl_allocBufferConst(sizeof(int) * 16, (void*)Surf::j);

	// Allocate buffers for each of the interesting points.  We don't know
	// how many there are initially, so must allocate more than enough space
	this->d_scale = cl_allocBuffer(initialPoints * sizeof(float));
	this->d_pixPos = cl_allocBuffer(initialPoints * sizeof(float2));
	this->d_laplacian = cl_allocBuffer(initialPoints * sizeof(int));

	// These buffers used to wait for the number of actual ipts to be known
	// before being allocated, instead now we'll only allocate them once
	// so that we can take advantage of optimized data transfers and reallocate
	// them if there's not enough space available
	this->d_length = cl_allocBuffer(initialPoints * DESC_SIZE * sizeof(float));
	this->d_desc = cl_allocBuffer(initialPoints * DESC_SIZE * sizeof(float));
	this->d_res = cl_allocBuffer(initialPoints * 109 * sizeof(float4));
	this->d_orientation = cl_allocBuffer(initialPoints * sizeof(float));

	// Allocate buffers to store the output data (descriptor information)
	// on the host
#ifdef OPTIMIZED_TRANSFERS
	this->h_scale = cl_allocBufferPinned(initialPoints * sizeof(float));
	this->h_pixPos = cl_allocBufferPinned(initialPoints * sizeof(float2));
	this->h_laplacian = cl_allocBufferPinned(initialPoints * sizeof(int));
	this->h_desc = cl_allocBufferPinned(initialPoints * DESC_SIZE * sizeof(float));
	this->h_orientation = cl_allocBufferPinned(initialPoints * sizeof(float));
#else
	this->scale = (float*)alloc(initialPoints * sizeof(float));
	this->pixPos = (float2*)alloc(initialPoints * sizeof(float2));
	this->laplacian = (int*)alloc(initialPoints * sizeof(int));
	this->desc = (float*)alloc(initialPoints * DESC_SIZE * sizeof(float));
	this->orientation = (float*)alloc(initialPoints * sizeof(float));
#endif
	// This is how much space is available for Ipts
	this->maxIpts = initialPoints;
}


//! Destructor
Surf::~Surf() {

	cl_freeMem(this->d_intImage);
	cl_freeMem(this->d_tmpIntImage);
	cl_freeMem(this->d_tmpIntImageT1);
	cl_freeMem(this->d_tmpIntImageT2);
	cl_freeMem(this->d_desc);
	cl_freeMem(this->d_orientation);
	cl_freeMem(this->d_gauss25);
	cl_freeMem(this->d_id);
	cl_freeMem(this->d_i);
	cl_freeMem(this->d_j);
	cl_freeMem(this->d_laplacian);
	cl_freeMem(this->d_pixPos);
	cl_freeMem(this->d_scale);
	cl_freeMem(this->d_res);
	cl_freeMem(this->d_length);

#ifdef OPTIMIZED_TRANSFERS
	cl_freeMem(this->h_orientation);
	cl_freeMem(this->h_scale);
	cl_freeMem(this->h_laplacian);
	cl_freeMem(this->h_desc);
	cl_freeMem(this->h_pixPos);
#else
	free(this->orientation);
	free(this->scale);
	free(this->laplacian);
	free(this->desc);
	free(this->pixPos);
#endif

	delete this->fh;
}

//! Computes the integral image of image img.
//! Assumes source image to be a  32-bit floating point.
/*!
Saves integral Image in d_intImage on the GPU
\param source Input Image as grabbed by OpenCv
*/
void Surf::computeIntegralImage(cv::Mat&  source)
{
	//! convert the image to single channel 32f

	// TODO This call takes about 4ms (is there any way to speed it up?)
	cv::Mat& img = getGray(source);

	// set up variables for data access
	int height = img.rows;
	int width = img.cols;
	float *data = (float*)img.data;

	cl_kernel scan_kernel;
	cl_kernel transpose_kernel;

	if (isUsingImages()) {
		// Copy the data to the GPU
		cl_copyImageToDevice(this->d_intImage, data, height, width);

		scan_kernel = this->kernel_list[KERNEL_SCANIMAGE];
		transpose_kernel = this->kernel_list[KERNEL_TRANSPOSEIMAGE];
	}
	else {
		// Copy the data to the GPU
		cl_copyBufferToDevice(this->d_intImage, data, sizeof(float)*width*height);

		// If it is possible to use the vector scan (scan4) use
		// it, otherwise, use the regular scan
		if (cl_deviceIsAMD() && width % 4 == 0 && height % 4 == 0)
		{
			// NOTE Change this to KERNEL_SCAN when running verification code.
			//      The reference code doesn't use a vector type and
			//      scan4 produces a slightly different integral image
			scan_kernel = this->kernel_list[KERNEL_SCAN4];
		}
		else
		{
			scan_kernel = this->kernel_list[KERNEL_SCAN];
		}
		transpose_kernel = this->kernel_list[KERNEL_TRANSPOSE];
	}

	// -----------------------------------------------------------------
	// Step 1: Perform integral summation on the rows
	// -----------------------------------------------------------------
	size_t localWorkSize1[2] = { 64, 1 };
	size_t globalWorkSize1[2] = { 64, height };
#ifdef _ORG_CL_DRIVER
	cl_setKernelArg(scan_kernel, 0, sizeof(cl_mem), (void *)&(this->d_intImage));
	cl_setKernelArg(scan_kernel, 1, sizeof(cl_mem), (void *)&(this->d_tmpIntImage));
	cl_setKernelArg(scan_kernel, 2, sizeof(int), (void *)&height);
	cl_setKernelArg(scan_kernel, 3, sizeof(int), (void *)&width);
	cl_executeKernel(scan_kernel, 2, globalWorkSize1, localWorkSize1, "Scan", 0);
#else
	coopcl_execute_scan(scan_kernel, 2, globalWorkSize1, localWorkSize1, this->d_intImage, this->d_tmpIntImage, height, width);
#endif 
	// -----------------------------------------------------------------
	// Step 2: Transpose
	// -----------------------------------------------------------------

	size_t localWorkSize2[] = { 16, 16 };
	size_t globalWorkSize2[] = { roundUp(width,16), roundUp(height,16) };
#ifdef _ORG_CL_DRIVER
	cl_setKernelArg(transpose_kernel, 0, sizeof(cl_mem), (void *)&(this->d_tmpIntImage));
	cl_setKernelArg(transpose_kernel, 1, sizeof(cl_mem), (void *)&(this->d_tmpIntImageT1));
	cl_setKernelArg(transpose_kernel, 2, sizeof(int), (void *)&height);
	cl_setKernelArg(transpose_kernel, 3, sizeof(int), (void *)&width);
	cl_executeKernel(transpose_kernel, 2, globalWorkSize2, localWorkSize2, "Transpose", 0);
#else
	coopcl_execute_transpose(transpose_kernel, 2, globalWorkSize2, localWorkSize2, this->d_tmpIntImage, this->d_tmpIntImageT1, height, width);
#endif

	// -----------------------------------------------------------------
	// Step 3: Run integral summation on the rows again (same as columns
	//         integral since we've transposed). 
	// -----------------------------------------------------------------

	int heightT = width;
	int widthT = height;

	size_t localWorkSize3[2] = { 64, 1 };
	size_t globalWorkSize3[2] = { 64, heightT };
#ifdef _ORG_CL_DRIVER
	cl_setKernelArg(scan_kernel, 0, sizeof(cl_mem), (void *)&(this->d_tmpIntImageT1));
	cl_setKernelArg(scan_kernel, 1, sizeof(cl_mem), (void *)&(this->d_tmpIntImageT2));
	cl_setKernelArg(scan_kernel, 2, sizeof(int), (void *)&heightT);
	cl_setKernelArg(scan_kernel, 3, sizeof(int), (void *)&widthT);
	cl_executeKernel(scan_kernel, 2, globalWorkSize3, localWorkSize3, "Scan", 1);
#else
	coopcl_execute_scan(scan_kernel, 2, globalWorkSize3, localWorkSize3, this->d_tmpIntImageT1, this->d_tmpIntImageT2, heightT, widthT);
#endif
	// -----------------------------------------------------------------
	// Step 4: Transpose back
	// -----------------------------------------------------------------

	size_t localWorkSize4[] = { 16, 16 };
	size_t globalWorkSize4[] = { roundUp(widthT,16), roundUp(heightT,16) };
#ifdef _ORG_CL_DRIVER
	cl_setKernelArg(transpose_kernel, 0, sizeof(cl_mem), (void *)&(this->d_tmpIntImageT2));
	cl_setKernelArg(transpose_kernel, 1, sizeof(cl_mem), (void *)&(this->d_intImage));
	cl_setKernelArg(transpose_kernel, 2, sizeof(int), (void *)&heightT);
	cl_setKernelArg(transpose_kernel, 3, sizeof(int), (void *)&widthT);
	cl_executeKernel(transpose_kernel, 2, globalWorkSize4, localWorkSize4, "Transpose", 1);
#else
	coopcl_execute_transpose(transpose_kernel, 2, globalWorkSize4, localWorkSize4, this->d_tmpIntImageT2, this->d_intImage, heightT, widthT);
#endif

	// release the gray image
	//cvReleaseImage(&img);
}


//! Create the SURF descriptors
/*!
Calculate orientation for all ipoints using the
sliding window technique from OpenSurf
\param d_intImage The integral image
\param width The width of the image
\param height The height of the image
*/
void Surf::createDescriptors(int i_width, int i_height)
{
	const size_t threadsPerWG = DESC_TH_WGS;
	const size_t wgsPerIpt = DESC_WGS_ITEM;

	cl_kernel surf64Descriptor_kernel = this->kernel_list[KERNEL_SURF_DESC];
	size_t localWorkSizeSurf64[2] = { threadsPerWG,1 };
	size_t globalWorkSizeSurf64[2] = { (wgsPerIpt*threadsPerWG),(size_t)numIpts };

#ifdef _ORG_CL_DRIVER
	cl_setKernelArg(surf64Descriptor_kernel, 0, sizeof(cl_mem), (void*)&(this->d_intImage));
	cl_setKernelArg(surf64Descriptor_kernel, 1, sizeof(int), (void*)&i_width);
	cl_setKernelArg(surf64Descriptor_kernel, 2, sizeof(int), (void*)&i_height);
	cl_setKernelArg(surf64Descriptor_kernel, 3, sizeof(cl_mem), (void*)&(this->d_scale));
	cl_setKernelArg(surf64Descriptor_kernel, 4, sizeof(cl_mem), (void*)&(this->d_desc));
	cl_setKernelArg(surf64Descriptor_kernel, 5, sizeof(cl_mem), (void*)&(this->d_pixPos));
	cl_setKernelArg(surf64Descriptor_kernel, 6, sizeof(cl_mem), (void*)&(this->d_orientation));
	cl_setKernelArg(surf64Descriptor_kernel, 7, sizeof(cl_mem), (void*)&(this->d_length));
	cl_setKernelArg(surf64Descriptor_kernel, 8, sizeof(cl_mem), (void*)&(this->d_j));
	cl_setKernelArg(surf64Descriptor_kernel, 9, sizeof(cl_mem), (void*)&(this->d_i));
	// Execute the descriptor kernel
	cl_executeKernel(surf64Descriptor_kernel, 2, globalWorkSizeSurf64, localWorkSizeSurf64, "CreateDescriptors");
#else
	coopcl_execute_descriptor(surf64Descriptor_kernel, 2, globalWorkSizeSurf64, localWorkSizeSurf64,
		this->d_intImage, i_width, i_height, d_scale, this->d_desc, d_pixPos, d_orientation,
		d_length, d_j, d_i);
#endif

	cl_kernel normSurf64_kernel = kernel_list[KERNEL_NORM_DESC];
	size_t localWorkSizeNorm64[] = { DESC_SIZE };
	size_t globallWorkSizeNorm64[] = { this->numIpts*DESC_SIZE };

#ifdef _ORG_CL_DRIVER
	cl_setKernelArg(normSurf64_kernel, 0, sizeof(cl_mem), (void*)&(this->d_desc));
	cl_setKernelArg(normSurf64_kernel, 1, sizeof(cl_mem), (void*)&(this->d_length));
	// Execute the descriptor normalization kernel
	cl_executeKernel(normSurf64_kernel, 1, globallWorkSizeNorm64, localWorkSizeNorm64, "NormalizeDescriptors");
#else

	coopcl_execute_descriptor_norm(normSurf64_kernel, 1, globallWorkSizeNorm64, localWorkSizeNorm64, d_desc, d_length);

#endif


}

//! Calculate orientation for all ipoints
/*!
Calculate orientation for all ipoints using the
sliding window technique from OpenSurf
\param i_width The image width
\param i_height The image height
*/
void Surf::getOrientations(int i_width, int i_height)
{
	cl_kernel getOrientation = this->kernel_list[KERNEL_GET_ORIENT1];
	cl_kernel getOrientation2 = this->kernel_list[KERNEL_GET_ORIENT2];

	size_t localWorkSize1[] = { ORIENT_WGS_STEP1 };
	size_t globalWorkSize1[] = { (size_t)this->numIpts*(size_t)ORIENT_WGS_STEP1 };

	/*!
	Assign the supplied Ipoint an orientation
	*/
#ifdef _ORG_CL_DRIVER
	cl_setKernelArg(getOrientation, 0, sizeof(cl_mem), (void *)&(this->d_intImage));
	cl_setKernelArg(getOrientation, 1, sizeof(cl_mem), (void *)&(this->d_scale));
	cl_setKernelArg(getOrientation, 2, sizeof(cl_mem), (void *)&(this->d_pixPos));
	cl_setKernelArg(getOrientation, 3, sizeof(cl_mem), (void *)&(this->d_gauss25));
	cl_setKernelArg(getOrientation, 4, sizeof(cl_mem), (void *)&(this->d_id));
	cl_setKernelArg(getOrientation, 5, sizeof(int), (void *)&i_width);
	cl_setKernelArg(getOrientation, 6, sizeof(int), (void *)&i_height);
	cl_setKernelArg(getOrientation, 7, sizeof(cl_mem), (void *)&(this->d_res));
	// Execute the kernel
	cl_executeKernel(getOrientation, 1, globalWorkSize1, localWorkSize1, "GetOrientations");
#else
	coopcl_execute_orientation1(getOrientation, 1, globalWorkSize1, localWorkSize1,
		this->d_intImage, this->d_scale, this->d_pixPos, this->d_gauss25, this->d_id, i_width, i_height, this->d_res);
#endif

	size_t localWorkSize2[] = { ORIENT_WGS_STEP2 };
	size_t globalWorkSize2[] = { (size_t)numIpts * (size_t)ORIENT_WGS_STEP2 };
#ifdef _ORG_CL_DRIVER
	cl_setKernelArg(getOrientation2, 0, sizeof(cl_mem), (void *)&(this->d_orientation));
	cl_setKernelArg(getOrientation2, 1, sizeof(cl_mem), (void *)&(this->d_res));
	// Execute the kernel
	cl_executeKernel(getOrientation2, 1, globalWorkSize2, localWorkSize2, "GetOrientations2");
#else
	coopcl_execute_orientation2(getOrientation2, 1, globalWorkSize2, localWorkSize2, d_orientation, d_res);
#endif
}

//! Allocates the memory objects required for the ipt descriptor information
void Surf::reallocateIptBuffers() {

	// Release the old memory objects (that were too small)
	cl_freeMem(d_scale);
	cl_freeMem(d_pixPos);
	cl_freeMem(d_laplacian);
	cl_freeMem(d_length);
	cl_freeMem(d_desc);
	cl_freeMem(d_res);
	cl_freeMem(d_orientation);

	free(this->orientation);
	free(this->scale);
	free(this->laplacian);
	free(this->desc);
	free(this->pixPos);

	int newSize = this->maxIpts;

	// Allocate new memory objects based on the new size
	this->d_scale = cl_allocBuffer(newSize * sizeof(float));
	this->d_pixPos = cl_allocBuffer(newSize * sizeof(float2));
	this->d_laplacian = cl_allocBuffer(newSize * sizeof(int));
	this->d_length = cl_allocBuffer(newSize * DESC_SIZE * sizeof(float));
	this->d_desc = cl_allocBuffer(newSize * DESC_SIZE * sizeof(float));
	this->d_res = cl_allocBuffer(newSize * 121 * sizeof(float4));
	this->d_orientation = cl_allocBuffer(newSize * sizeof(float));

#ifdef OPTIMIZED_TRANSFERS
	this->h_scale = cl_allocBufferPinned(newSize * sizeof(float));
	this->h_pixPos = cl_allocBufferPinned(newSize * sizeof(float2));
	this->h_laplacian = cl_allocBufferPinned(newSize * sizeof(int));
	this->h_desc = cl_allocBufferPinned(newSize * DESC_SIZE * sizeof(float));
	this->h_orientation = cl_allocBufferPinned(newSize * sizeof(float));
#else
	this->scale = (float*)alloc(newSize * sizeof(float));
	this->pixPos = (float2*)alloc(newSize * sizeof(float2));
	this->laplacian = (int*)alloc(newSize * sizeof(int));
	this->desc = (float*)alloc(newSize * DESC_SIZE * sizeof(float));
	this->orientation = (float*)alloc(newSize * sizeof(float));
#endif
}


//! This function gets called each time SURF is run on a new frame.  It prevents
//! having to create and destroy the object each time (lots of OpenCL overhead)
void Surf::reset()
{
	this->fh->reset();
}


//! Retreive the descriptors from the GPU
/*!
Copy data back from the GPU into an IpVec structure on the host
*/
IpVec* Surf::retrieveDescriptors()
{
	IpVec* ipts = new IpVec();

	if (this->numIpts == 0)
	{
		return ipts;
	}

	// Copy back the output data

#ifdef OPTIMIZED_TRANSFERS
	// We're using pinned memory for the transfers.  The data is 
	// copied back to pinned memory and then must be mapped before
	// it's usable on the host

	// Copy back Laplacian data
	this->laplacian = (int*)cl_copyAndMapBuffer(this->h_laplacian,
		this->d_laplacian, this->numIpts * sizeof(int));

	// Copy back scale data
	this->scale = (float*)cl_copyAndMapBuffer(this->h_scale,
		this->d_scale, this->numIpts * sizeof(float));

	// Copy back pixel positions
	this->pixPos = (float2*)cl_copyAndMapBuffer(this->h_pixPos,
		this->d_pixPos, this->numIpts * sizeof(float2));

	// Copy back descriptors
	this->desc = (float*)cl_copyAndMapBuffer(this->h_desc,
		this->d_desc, this->numIpts * DESC_SIZE * sizeof(float));

	// Copy back orientation data
	this->orientation = (float*)cl_copyAndMapBuffer(this->h_orientation,
		this->d_orientation, this->numIpts * sizeof(float));
#else
	// Copy back Laplacian information
	cl_copyBufferToHost(this->laplacian, this->d_laplacian,
		(this->numIpts) * sizeof(int), CL_FALSE);

	// Copy back scale data
	cl_copyBufferToHost(this->scale, this->d_scale,
		(this->numIpts) * sizeof(float), CL_FALSE);

	// Copy back pixel positions
	cl_copyBufferToHost(this->pixPos, this->d_pixPos,
		(this->numIpts) * sizeof(float2), CL_FALSE);

	// Copy back descriptors
	cl_copyBufferToHost(this->desc, this->d_desc,
		(this->numIpts)*DESC_SIZE * sizeof(float), CL_FALSE);

	// Copy back orientation data
	cl_copyBufferToHost(this->orientation, this->d_orientation,
		(this->numIpts) * sizeof(float), CL_TRUE);
#endif  

	// Parse the data into Ipoint structures
	for (int i = 0; i<(this->numIpts); i++)
	{
		Ipoint ipt;
		ipt.x = pixPos[i].x;
		ipt.y = pixPos[i].y;
		ipt.scale = scale[i];
		ipt.laplacian = laplacian[i];
		ipt.orientation = orientation[i];
		memcpy(ipt.descriptor, &desc[i * 64], sizeof(float) * 64);
		ipts->push_back(ipt);
	}

#ifdef OPTIMIZED_TRANSFERS
	// We're done reading from the buffers, so we unmap
	// them so they can be used again by the device
	cl_unmapBuffer(this->h_laplacian, this->laplacian);
	cl_unmapBuffer(this->h_scale, this->scale);
	cl_unmapBuffer(this->h_pixPos, this->pixPos);
	cl_unmapBuffer(this->h_desc, this->desc);
	cl_unmapBuffer(this->h_orientation, this->orientation);
#endif

	return ipts;
}


//! Function that builds vector of interest points.  This is the main SURF function
//! that will be called for any type of input.
/*!
High level driver function for entire OpenSurfOpenCl
\param img image to find Ipoints within
\param upright Switch for future functionality of upright surf
\param fh FastHessian object
*/
void Surf::run(cv::Mat&  img, bool upright)
{

	if (upright)
	{
		// Extract upright (i.e. not rotation invariant) descriptors
		printf("Upright surf not supported\n");
		exit(1);
	}



	// Perform the scan sum of the image (populates d_intImage)
	// GPU kernels: scan (x2), tranpose (x2)
	this->computeIntegralImage(img);

	// Determines the points of interest
	// GPU kernels: init_det, hessian_det (x12), non_max_suppression (x3)
	// GPU mem transfer: copies back the number of ipoints 
	this->numIpts = this->fh->getIpoints(img, this->d_intImage, this->d_laplacian,
		this->d_pixPos, this->d_scale, this->maxIpts);

	// Verify that there was enough space allocated for the number of
	// Ipoints found
	if (this->numIpts >= this->maxIpts) {
		// If not enough space existed, we need to reallocate space and
		// run the kernels again

		printf("Not enough space for Ipoints, reallocating and running again\n");
		this->maxIpts = this->numIpts * 2;
		this->reallocateIptBuffers();
		// XXX This was breaking sometimes
		this->fh->reset();
		this->numIpts = fh->getIpoints(img, this->d_intImage,
			this->d_laplacian, this->d_pixPos, this->d_scale, this->maxIpts);
	}

	printf("There were %d interest points\n", this->numIpts);

	// Main SURF-64 loop assigns orientations and gets descriptors    
	if (this->numIpts == 0) return;

	// GPU kernel: getOrientation1 (1x), getOrientation2 (1x)
	this->getOrientations(img.cols, img.rows);

	// GPU kernel: surf64descriptor (1x), norm64descriptor (1x)
	this->createDescriptors(img.cols, img.rows);
}


#endif
