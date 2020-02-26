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

#include "nearestNeighbor.h"
#include "utils.h"

std::vector<distPoint>* findNearestNeighbors(
	IpVec &ipts1, IpVec &ipts2,
	cl_kernel* kernel_list) 
{
    if(!ipts1.size() || !ipts2.size()) {
	    std::vector<distPoint>* dps = new std::vector<distPoint>(0);
	    return dps;
    }

    cl_kernel NN_kernel = kernel_list[KERNEL_NN];

    // Set up memory on device and send ipts data to device
    // copy ipts<1,2> to device
    // also need to alloate memory for the distancePoints
    cl_mem d_ipts1, d_ipts2;
    cl_mem d_distancePoints, d_tempDistPoints;

    size_t numIpts1 = ipts1.size();
    size_t numIpts2 = ipts2.size();

    // Allocate some GPU memory for Ipoints and distance points 
    d_ipts1 = cl_allocBuffer(numIpts1*sizeof(Ipoint), CL_MEM_READ_ONLY);
    d_ipts2 = cl_allocBuffer(numIpts2*sizeof(Ipoint), CL_MEM_READ_ONLY);
    d_distancePoints = cl_allocBuffer(numIpts1*sizeof(distPoint), CL_MEM_READ_ONLY);
    d_tempDistPoints = cl_allocBuffer(64*sizeof(float)*numIpts1);

    // Copy input Ipoints to device
    cl_copyBufferToDevice(d_ipts1, &ipts1[0], numIpts1*sizeof(Ipoint));
    cl_copyBufferToDevice(d_ipts2, &ipts2[0], numIpts2*sizeof(Ipoint));

    // Set kernel arguments
    unsigned int ipts2Size = (unsigned int)ipts2.size();

    cl_setKernelArg(NN_kernel, 0, sizeof(cl_mem), (void *)&d_ipts1);
    cl_setKernelArg(NN_kernel, 1, sizeof(cl_mem), (void *)&d_ipts2);
    cl_setKernelArg(NN_kernel, 2, sizeof(cl_mem), (void *)&d_distancePoints);
    cl_setKernelArg(NN_kernel, 3, 64*sizeof(float), NULL);
    cl_setKernelArg(NN_kernel, 4, sizeof(unsigned int), (void *)&ipts2Size);
    // Enqueue the kernel
    size_t localWorkSize[1];
    size_t globalWorkSize[1];
    globalWorkSize[0] = ipts1.size()*64;
    localWorkSize[0] = 64;
    // Run the nearest neighbor kernel
    cl_executeKernel(NN_kernel, 1, globalWorkSize, localWorkSize,"NearestNeighbor");

    // Transfer data back from the device to a temporary location
    distPoint* tmpDistPts = (distPoint*)alloc(numIpts1*sizeof(distPoint));
    cl_copyBufferToHost(tmpDistPts, d_distancePoints, 
	    numIpts1*sizeof(distPoint));

    // Copy the distance points into a vector
    std::vector<distPoint>* distancePoints = 
        new std::vector<distPoint>(tmpDistPts, tmpDistPts + numIpts1);

    // Release the temp buffer
    free(tmpDistPts);

    // Release device buffers
    cl_freeMem(d_ipts1);
    cl_freeMem(d_ipts2);
    cl_freeMem(d_distancePoints);
    cl_freeMem(d_tempDistPoints);

    // Sort the distancePoints so that the smallest distance is at the front
    std::sort(distancePoints->begin(),distancePoints->end(),distPointsCmp());

    // Make a histogram of the occurance of each point (to mark duplicates)
    std::vector<int> dupCount(ipts2.size(), 0);
    for(unsigned int i = 0; i < distancePoints->size(); i++){

        if(dupCount[distancePoints->at(i).point2] > 0) {
            // This point has already been used (and has a better match),
            // mark as duplicate
            distancePoints->at(i).dup = true;
        }
        else {
            distancePoints->at(i).dup = false;
        }
	    dupCount[distancePoints->at(i).point2]++;

        // Prior to this, matches were not determined by the
        // minimum distance, this metric changes that. 
        // Note that 0.035 is arbitrary (based on observed values)
        // and may need to be fine tuned.
        if(distancePoints->at(i).dist < 0.035) {
            distancePoints->at(i).match = 1;
        }
        else {
            distancePoints->at(i).match = 0;
        }
    }

    return distancePoints;
}