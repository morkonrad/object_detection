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

#define WG_SIZE 64
#define HALF_WG_SIZE (WG_SIZE/2)

/*
 * This kernel does a prefix scan on any size image.  Each work group marches down a row
 * of the image performing a sub-scan on WG_SIZE elements at a time.
 * The value obtained by the thread with the biggest ID is kept as the base for the
 * next sub-scan.
 */
__kernel
void scan(__global float *input, 
          __global float *output, 
                     int  rows,
                     int  cols) {

    // Each work group is responsible for scanning a row

    // If we add additional elements to local memory (half the
    // work group size) and fill them with zeros, we don't need
    // conditionals in the code to check for out-of-bounds
    // accesses
    __local float lData[WG_SIZE+HALF_WG_SIZE];
    __local float lData2[WG_SIZE+HALF_WG_SIZE];

    int myGlobalY = get_global_id(1);
    int myLocalX = get_local_id(0);

    // Initialize out-of-bounds elements with zeros
    if(myLocalX < HALF_WG_SIZE) {
        lData[myLocalX] = 0.0f;
        lData2[myLocalX] = 0.0f;
    }
    
    // Start past out-of-bounds elements
    myLocalX += HALF_WG_SIZE;
    
    // This value needs to be added to the local data.  It's the highest
    // value from the prefix scan of the previous elements in the row
    float prevMaxVal = 0;

    // March down a row WG_SIZE elements at a time
    int iterations = cols/WG_SIZE;
    if(cols % WG_SIZE != 0) 
    {
        // If cols are not an exact multiple of WG_SIZE, then we need to do
        // one more iteration 
        iterations++;
    }
   
    for(int i = 0; i < iterations; i++) {
                
        int columnOffset = i*WG_SIZE + get_local_id(0);

        // Don't do anything if a thread's index is past the end of the row.  We still need
        // the thread available for memory barriers though.
        if(columnOffset < cols) {
            // Cache the input data in local memory
            lData[myLocalX] = input[myGlobalY*cols + columnOffset];
        }
        else {
            // Otherwise just store zeros 
            lData[myLocalX] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // 1
        lData2[myLocalX] = lData[myLocalX] + lData[myLocalX-1];
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // 2
        lData[myLocalX] = lData2[myLocalX] + lData2[myLocalX-2];
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // 4
        lData2[myLocalX] = lData[myLocalX] + lData[myLocalX-4];
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // 8
        lData[myLocalX] = lData2[myLocalX] + lData2[myLocalX-8];
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // 16
        lData2[myLocalX] = lData[myLocalX] + lData[myLocalX-16];
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // 32
        lData[myLocalX] = lData2[myLocalX] + lData2[myLocalX-32];
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Write data out to global memory
        if(columnOffset < cols) {
            output[myGlobalY*cols + columnOffset] = lData[myLocalX] + prevMaxVal;
        }
        
        // Copy the value from the highest thread in the group to my local index
        prevMaxVal += lData[WG_SIZE+HALF_WG_SIZE-1];
    }
}


__kernel
void scan4(__global float4 *input, 
           __global float4 *output, 
                       int  rows,
                       int  cols) {

    // Each work group is responsible for scanning a row

    // If we add additional elements to local memory (half the
    // work group size) and fill them with zeros, we don't need
    // conditionals in the code to check for out-of-bounds
    // accesses
    __local float4 lData[WG_SIZE+HALF_WG_SIZE];
    __local float4 lData2[WG_SIZE+HALF_WG_SIZE];

    int myGlobalY = get_global_id(1);
    int myLocalX = get_local_id(0);

    // Initialize out-of-bounds elements with zeros
    if(myLocalX < HALF_WG_SIZE) {
        lData[myLocalX] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        lData2[myLocalX] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    // Start past out-of-bounds elements
    myLocalX += HALF_WG_SIZE;
    
    // This value needs to be added to the local data.  It's the highest
    // value from the prefix scan of the previous elements in the row
    float prevMaxVal = 0;
    
    // There will actually be only 1/4 of the columns since we've changed
    // the data type to float4
    int columns = cols/4;

    // March down a row WG_SIZE elements at a time
    int iterations = columns/WG_SIZE;
    if(columns % WG_SIZE != 0) 
    {
        // If cols are not an exact multiple of WG_SIZE, then we need to do
        // one more iteration 
        iterations++;
    }
   
    for(int i = 0; i < iterations; i++) {
                
        int columnOffset = i*WG_SIZE + get_local_id(0);

        // Don't do anything if a thread's index is past the end of the row.  We still need
        // the thread available for memory barriers though.
        if(columnOffset < columns) {
            // Cache the input data in local memory
            lData[myLocalX] = input[myGlobalY*columns + columnOffset];
        }
        else {
            // Otherwise just store zeros 
            lData[myLocalX] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
        
        lData[myLocalX].y += lData[myLocalX].x;
        lData[myLocalX].z += lData[myLocalX].y;
        lData[myLocalX].w += lData[myLocalX].z;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // 1
        lData2[myLocalX] = lData[myLocalX] + lData[myLocalX-1].w;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // 2
        lData[myLocalX] = lData2[myLocalX] + lData2[myLocalX-2].w;
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // 4
        lData2[myLocalX] = lData[myLocalX] + lData[myLocalX-4].w;
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // 8
        lData[myLocalX] = lData2[myLocalX] + lData2[myLocalX-8].w;
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // 16
        lData2[myLocalX] = lData[myLocalX] + lData[myLocalX-16].w;
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // 32
        lData[myLocalX] = lData2[myLocalX] + lData2[myLocalX-32].w;
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Write data out to global memory
        if(columnOffset < columns) {
            output[myGlobalY*columns + columnOffset] = lData[myLocalX] + prevMaxVal;
        }
        
        // Copy the value from the highest thread in the group to my local index
        prevMaxVal += lData[WG_SIZE+HALF_WG_SIZE-1].w;
    }
}


__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP           |  
                               CLK_FILTER_NEAREST;
                               
__kernel
void scanImage(__read_only image2d_t input, 
               __write_only image2d_t output, 
                     int  rows,
                     int  cols) {

    // Each work group is responsible for scanning a row

    // If we add additional elements to local memory (half the
    // work group size) and fill them with zeros, we don't need
    // conditionals in the code to check for out-of-bounds
    // accesses
    __local float lData[WG_SIZE+HALF_WG_SIZE];
    __local float lData2[WG_SIZE+HALF_WG_SIZE];

    int myGlobalY = get_global_id(1);
    int myLocalX = get_local_id(0);

    // Initialize out-of-bounds elements with zeros
    if(myLocalX < HALF_WG_SIZE) {
        lData[myLocalX] = 0.0f;
        lData2[myLocalX] = 0.0f;
    }
    
    // Start past out-of-bounds elements
    myLocalX += HALF_WG_SIZE;
    
    // This value needs to be added to the local data.  It's the highest
    // value from the prefix scan of the previous elements in the row
    float prevMaxVal = 0;

    // March down a row WG_SIZE elements at a time
    int iterations = cols/WG_SIZE;
    if(cols % WG_SIZE != 0) 
    {
        // If cols are not an exact multiple of WG_SIZE, then we need to do
        // one more iteration 
        iterations++;
    }
   
    for(int i = 0; i < iterations; i++) {
                
        int columnOffset = i*WG_SIZE + get_local_id(0);

        // Don't do anything if a thread's index is past the end of the row.  We still need
        // the thread available for memory barriers though.
        
        // Cache the input data in local memory
        lData[myLocalX] = read_imagef(input, sampler, (int2)(columnOffset,myGlobalY)).x;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // 1
        lData2[myLocalX] = lData[myLocalX] + lData[myLocalX-1];
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // 2
        lData[myLocalX] = lData2[myLocalX] + lData2[myLocalX-2];
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // 4
        lData2[myLocalX] = lData[myLocalX] + lData[myLocalX-4];
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // 8
        lData[myLocalX] = lData2[myLocalX] + lData2[myLocalX-8];
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // 16
        lData2[myLocalX] = lData[myLocalX] + lData[myLocalX-16];
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // 32
        lData[myLocalX] = lData2[myLocalX] + lData2[myLocalX-32];
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Write data out to global memory
        if(columnOffset < cols) {
            write_imagef(output, (int2)(columnOffset, myGlobalY), (float4)(lData[myLocalX] + prevMaxVal, 0.0f, 0.0f, 0.0f));
        }
        
        // Copy the value from the highest thread in the group to my local index
        prevMaxVal += lData[WG_SIZE+HALF_WG_SIZE-1];
    }
}


__kernel
void transpose(__global float* iImage,
               __global float* oImage,
                        int    inRows,
                        int    inCols) {
                        
    __local float tmpBuffer[256];

    // Work groups will perform the transpose (i.e., an entire work group will be moved from
    // one part of the image to another) 
    //int myWgInX = get_group_id(0);
	int myWgInX = ccl_get_group_id(0);
    int myWgInY = get_group_id(1);

    // The local index of a thread should not change (threads with adjacent IDs 
    // within a work group should always perform adjacent memory accesses).  We will 
    // account for the clever indexing of local memory later.
    int myLocalX = get_local_id(0);
    int myLocalY = get_local_id(1);
    
    int myGlobalInX = myWgInX*16 + myLocalX;
    int myGlobalInY = myWgInY*16 + myLocalY;

    // Don't read out of bounds
    if(myGlobalInX < inCols && myGlobalInY < inRows) {
        
        // Cache data to local memory (coalesced reads)
        tmpBuffer[myLocalY*16 + myLocalX] = iImage[myGlobalInY*inCols + myGlobalInX];

    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // This avoids some confusion with dimensions, but otherwise aren't needed
    int outRows = inCols;
    int outCols = inRows;

    // Swap work group IDs for their location after the transpose
    int myWgOutX = myWgInY;
    int myWgOutY = myWgInX;

    int myGlobalOutX = myWgOutX*16 + myLocalX;
    int myGlobalOutY = myWgOutY*16 + myLocalY;

    // When writing back to global memory, we need to swap image dimensions (for the
    // transposed size), and also need to swap thread X/Y indices in local memory
    // (to achieve coalesced memory writes)

    // Don't write out of bounds
    if(myGlobalOutX >= 0 && myGlobalOutX < outCols && 
       myGlobalOutY >= 0 && myGlobalOutY < outRows) {

       // The read from tmpBuffer is going to conflict, but the write should be coalesced
       oImage[myGlobalOutY*outCols + myGlobalOutX] = tmpBuffer[myLocalX*16 + myLocalY]; 
    }

    return;
} 


__kernel
void transposeImage(__read_only image2d_t iImage, 
                   __write_only image2d_t oImage,
                                      int inRows,
                                      int inCols) {
                        
    __local float tmpBuffer[256];

    // Work groups will perform the transpose (i.e., an entire work group will be moved from
    // one part of the image to another) 
    int myWgInX = get_group_id(0);
    int myWgInY = get_group_id(1);

    // The local index of a thread should not change (threads with adjacent IDs 
    // within a work group should always perform adjacent memory accesses).  We will 
    // account for the clever indexing of local memory later.
    int myLocalX = get_local_id(0);
    int myLocalY = get_local_id(1);
    
    int myGlobalInX = myWgInX*16 + myLocalX;
    int myGlobalInY = myWgInY*16 + myLocalY;

    // Cache data to local memory (coalesced reads)
    tmpBuffer[myLocalY*16 + myLocalX] = read_imagef(iImage, sampler, (int2)(myGlobalInX, myGlobalInY)).x;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // This avoids some confusion with dimensions, but otherwise aren't needed
    int outRows = inCols;
    int outCols = inRows;

    // Swap work group IDs for their location after the transpose
    int myWgOutX = myWgInY;
    int myWgOutY = myWgInX;

    int myGlobalOutX = myWgOutX*16 + myLocalX;
    int myGlobalOutY = myWgOutY*16 + myLocalY;

    // When writing back to global memory, we need to swap image dimensions (for the
    // transposed size), and also need to swap thread X/Y indices in local memory
    // (to achieve coalesced memory writes)

    // Don't write out of bounds
    if(myGlobalOutX >= 0 && myGlobalOutX < outCols && 
       myGlobalOutY >= 0 && myGlobalOutY < outRows) {
       // The read from tmpBuffer is going to conflict, but the write should be coalesced
       write_imagef(oImage, (int2)(myGlobalOutX, myGlobalOutY), (float4)(tmpBuffer[myLocalX*16 + myLocalY], 0.0f, 0.0f, 0.0f));
    }
} 