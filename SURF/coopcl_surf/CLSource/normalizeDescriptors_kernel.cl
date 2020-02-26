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

int ccl_get_group_id(const int id) { return (get_global_id(id)-get_local_id(id))/get_local_size(id); }

__kernel void
normalizeDescriptors(__global float* surfDescriptors, 
                     __global float* descLengths)
{
    // Previous kernels have computed 64 descriptors (surfDescriptors) and 
    // 16 lengths (descLengths) for each interesting point found by SURF.
    // This kernel will sum up all of the lengths for each interesting point 
    // and take the square root.  This value will then be used to scale the 
    // descriptors.

    // Note that each work group contains 64 work items (one per descriptor).

    // This array is used to cache data that will be accessed multiple times.  
    // Since it is declared __local, only one instance is created and shared 
    // by the entire work group
    __local float ldescLengths[16];

    // Get the offset for the descriptor that this work group will normalize
    //int descOffset = get_group_id(0) * 64;
	int descOffset = ccl_get_group_id(0) * 64;

    // Get the offset for the descriptor lengths that this work group will 
    // use to scale the descriptors
    //int lenOffset = get_group_id(0) * 16;
	int lenOffset = ccl_get_group_id(0) * 16;

    // Get this work item’s ID within the work group
    int tid = get_local_id(0);

    // Only 16 of the work items are needed to cache the lengths
    if(tid < 16) {
        // Have the first 16 work items cache the lengths for this 
        // point in local memory
        ldescLengths[tid] = descLengths[lenOffset + tid];
    } 

    barrier(CLK_LOCAL_MEM_FENCE);

    // Work items work together to perform a parallel reduction of the 
    // descriptor lengths (result gets stored in ldescLength[0])
    for(int i = 8; i > 0; i >>= 1)    
    {                
        if (tid < i)     
        {                    
            ldescLengths[tid] += ldescLengths[tid + i];
         
        }             
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate the normalized length of the descriptors
    float lengthOfDescriptor = 1.0f/sqrt(ldescLengths[0]);

    // Scale each descriptor
    surfDescriptors[descOffset + tid] *= lengthOfDescriptor;	   
}