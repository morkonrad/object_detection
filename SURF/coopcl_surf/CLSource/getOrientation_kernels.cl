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

#ifdef M_PI_F
#define pi M_PI_F
#else 
#define pi 3.141592654f
#endif

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#ifdef IMAGES_SUPPORTED
// CLK_ADDRESS_CLAMP returns (0,0,0,1) for out of bounds accesses
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE   |  
                               CLK_FILTER_NEAREST;
#endif

//! Calculate the value of the 2d gaussian at x,y
float gaussian(float x, float y, float sig)
{
    return (1.0f/(2.0f*pi*sig*sig)) * exp(-(x*x+y*y)/(2.0f*sig*sig));
}

//! Calculate the integral of a region
float BoxIntegral(
#ifdef IMAGES_SUPPORTED
                  __read_only image2d_t data, 
#else
                  __global float* data,
#endif
                  int width, int height, int row, int col, 
                  int rows, int cols) 
{

    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    // The subtraction by one for row/col is because row/col is inclusive.
    int r1 = min(row, height) - 1;
    int c1 = min(col, width)  - 1;
    int r2 = min(row + rows, height) - 1;
    int c2 = min(col + cols, width)  - 1;

#ifdef IMAGES_SUPPORTED
    A = read_imagef(data, sampler, (int2)(c1, r1)).x;
    B = read_imagef(data, sampler, (int2)(c2, r1)).x;
    C = read_imagef(data, sampler, (int2)(c1, r2)).x;
    D = read_imagef(data, sampler, (int2)(c2, r2)).x;
#else    
    if (r1 >= 0 && c1 >= 0) A = data[r1 * width + c1];  
    if (r1 >= 0 && c2 >= 0) B = data[r1 * width + c2];  
    if (r2 >= 0 && c1 >= 0) C = data[r2 * width + c1];
    if (r2 >= 0 && c2 >= 0) D = data[r2 * width + c2];
#endif 

    return max(0.0f, A - B - C + D);
}


//! Calculate Haar wavelet responses in X direction
float haarX(
#ifdef IMAGES_SUPPORTED
            __read_only image2d_t img, 
#else
            __global float* img,
#endif
            int width, int height, int row, int column, int s)
{
    return BoxIntegral(img, width, height, row-s/2, column,     s, s/2) -
           BoxIntegral(img, width, height, row-s/2, column-s/2, s, s/2);
}


//! Calculate Haar wavelet responses in Y direction
float haarY(
#ifdef IMAGES_SUPPORTED
            __read_only image2d_t img, 
#else
            __global float* img,
#endif
            int width, int height, int row, int column, int s)
{
    return BoxIntegral(img, width, height, row,     column-s/2, s/2, s) -
           BoxIntegral(img, width, height, row-s/2, column-s/2, s/2, s);
}


//! Get the angle from the +ve x-axis of the vector given by (X Y)
float getAngle(float X, float Y)
{
    if(X >= 0 && Y >= 0)
        return atan(Y/X);

    if(X < 0 && Y >= 0)
        return pi - atan(-Y/X);

    if(X < 0 && Y < 0)
        return pi + atan(Y/X);

    if(X >= 0 && Y < 0)
        return 2*pi - atan(-Y/X);

    return 0;
}


//! 
__kernel void 
getOrientationStep1(
#ifdef IMAGES_SUPPORTED
                    __read_only image2d_t d_img, 
#else
                    __global float* d_img, 
#endif
                    __global float* d_scale,  
                    __global float2* d_pixPos, 
                    __global float* d_gauss25,
                    __global unsigned int* d_id,
                    int i_width, 
                    int i_height,
                    __global float4* res)
{

     // Cache the gaussian data in local memory
    __local float l_gauss25[49];
    __local unsigned int l_id[13];
    
    int localId = get_local_id(0);
    //int groupId = get_group_id(0);
	int groupId = ccl_get_group_id(0);
    
    int i = (int)(localId/13) - 6;
    int j = (localId%13) - 6;
   
    // Buffer gauss data in local memory
    if(localId < 49) 
    {
        l_gauss25[localId] = d_gauss25[localId];
    }
    // Buffer d_ids in local memory
    if(localId < 13) 
    {
        l_id[localId] = d_id[localId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // NOTE Since this only reads once from global memory, it is
    //      quicker just to leave the data there.  There is a higher
    //      overhead from placing it in constant memory.
    int s = round(d_scale[groupId]);
    int r = round(d_pixPos[groupId].y);
    int c = round(d_pixPos[groupId].x);

    float gauss = 0.f;
    float4 rs = {0.0f, 0.0f, FLT_MAX, 1.0f};
    
    __local int angleCount[1];
    if(localId == 0) {
        angleCount[0] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate haar responses for points within radius of 6*scale
    if(i*i + j*j < 36)
    {
        gauss = l_gauss25[7*l_id[i+6]+l_id[j+6]];
        rs.x = gauss * haarX(d_img, i_width, i_height, r+j*s, c+i*s, 4*s);
        rs.y = gauss * haarY(d_img, i_width, i_height, r+j*s, c+i*s, 4*s);
        rs.z = getAngle(rs.x, rs.y);
        int index = atom_add(&angleCount[0], 1);
        
        res[groupId * 109 +  index] = rs;
    }  
}

float magnitude(float2 val) {

    return (val.x*val.x + val.y*val.y);
}
 
//! 
__kernel void getOrientationStep2(__global float *d_orientation, 
                                  __global float4 *d_res)
{

    // There are 42 threads
    int tid = get_local_id(0);
    //int groupId = get_group_id(0);
	int groupId = ccl_get_group_id(0);

    // calculate the dominant direction
    float ang1= 0.15f * (float)tid;
    float ang2 = (ang1+(pi/3.0f) > 2*pi ? ang1-(5.0f*pi/3.0f) : ang1+(pi/3.0f));

    __local float4 res[109];

    // Cache data in local memory
    for(uint k = tid; k < 109; k += get_local_size(0)) 
    {
        res[k] = d_res[groupId * 109 + k];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
 
    // loop slides pi/3 window around feature point
    
    __local float2 sum[42];

    sum[tid] = (float2)(0.f, 0.f);

    // If tid is 0-34, ang1 < ang2
    if(tid <= 34) {
        for(uint k = 0; k < 109; ++k)
        {	
            const float4 rs = res[k];	
            // get angle from the x-axis of the sample point	
            const float ang = rs.z;

            // determine whether the point is within the window
            int check = ang1 < ang && ang < ang2;
            sum[tid].x += rs.x * check;
            sum[tid].y += rs.y * check;
        }
    }
    else {
        // If tid is 35-41, ang2 < ang1
        for(uint k = 0; k < 109; ++k)
        {	
            const float4 rs = res[k];	
            // get angle from the x-axis of the sample point	
            const float ang = rs.z;

            // determine whether the point is within the window
            int check = ((ang > 0 && ang < ang2) || (ang > ang1 && ang < 2*pi));
            sum[tid].x += rs.x * check;
            sum[tid].y += rs.y * check;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE); 
    
    // If the vector produced from this window is longer than all
    // previous vectors then this forms the new dominant direction

    for(int offset = 32; offset > 0; offset >>= 1) 
    {
        if (tid < offset && tid + offset < 42) {

            if(magnitude(sum[tid]) < magnitude(sum[tid+offset])) {
                
                sum[tid] = sum[tid+offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); 
    }
    
    if (tid == 0) 
    {
        // assign orientation of the dominant response vector
        d_orientation[groupId] = getAngle(sum[0].x, sum[0].y);
    }    
}
