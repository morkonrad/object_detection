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

#define DES_THREADS 81

#ifdef M_PI_F
#define pi M_PI_F
#else
#define pi 3.141592654f
#endif

#ifdef IMAGES_SUPPORTED
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP           |
                               CLK_FILTER_NEAREST;
#endif

float 
BoxIntegral( 
#ifdef IMAGES_SUPPORTED
              __read_only image2d_t data,
#else              
              __global float* data, 
#endif
              int width, int height, int row, int col, int rows, int cols) 
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

    return max(0.f, A - B - C + D);
}


//! Calculate Haar wavelet responses in x direction
float haarX(
#ifdef IMAGES_SUPPORTED
              __read_only image2d_t img,
#else              
              __global float* img, 
#endif
              int width, int height, int row, int column, int s)
{
    return BoxIntegral(img, width, height, row-s/2, column, s, s/2) -
           BoxIntegral(img, width, height, row-s/2, column-s/2, s, s/2);
}


//! Calculate Haar wavelet responses in y direction
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


void sumDesc(__local float4* desc, int tha, int length) 
{
    // do one loop to get all the > tha 64 values
    if(tha < length - 64) 
    {
        desc[tha] += desc[tha + 64];	
    }
    barrier(CLK_LOCAL_MEM_FENCE); 

    int stride ; // 64/2
    for (stride = 32; stride>0; stride>>=1)
    {
        if (tha < stride)	
        {
            desc[tha] += desc[tha + stride];			  
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

//! Calculate the value of the 2d gaussian at x,y
float gaussian(float x, float y, float sig)
{
    return 1.0f/(2.0f*pi*sig*sig) * exp(-(x*x+y*y)/(2.0f*sig*sig));
}


__kernel void createDescriptors(
#ifdef IMAGES_SUPPORTED
              __read_only image2d_t intImage,
#else              
              __global float* intImage, 
#endif
              int width, int height, 
              __global float* scale, 
              __global float4* surfDescriptor, 
              __global float2* pos, 
              __global float* orientation,
              __global float* descLength,
              __constant int* mj,
              __constant int* mi)
{
    __local float4 desc[DES_THREADS];

    // There are 16 work groups per descriptor.  The groups are arranged as 
    // 16 x NumDescriptors, so each bIdy is a new descriptor.
    
	//int bIdx = get_group_id(0);
	int bIdx = ccl_get_group_id(0);
    
	//int bIdy = get_group_id(1);
	int bIdy = ccl_get_group_id(1);
  
    // get the x & y indexes and absolute
    int thx = get_local_id(0);
    int tha = get_local_id(1) * get_local_size(0) + thx;

    // init shared memory to zero
    desc[tha] =(float4)(0.0f,0.0f,0.0f,0.0f);

    // The 16 work groups are logically arranged in a 4x4 square
    float cx = 0.5f, cy = 0.5f; //Subregion centers for the 4x4 gaussian weighting
    cx += (float)((int)(bIdx/4));
    cy += (float)((int)(bIdx%4));
    
    //const int mj[16] = {	
    //           -12, -7, -2, 3,
    //			 -12, -7, -2, 3,
    //			 -12, -7, -2, 3,
    //			 -12, -7, -2, 3};

    //const int mi[16] = { 	
    //          -12,-12,-12,-12,
    //		     -7, -7, -7, -7, 
    //		     -2, -2, -2, -2, 
    //		      3,  3,  3,  3};

    int j = mj[bIdx];
    int i = mi[bIdx];

    float x = round(pos[bIdy].x);
    float y = round(pos[bIdy].y);

    float orient = orientation[bIdy];
    float co = cos(orient);
    float si = sin(orient);
    
    float thScale = scale[bIdy];

    float ix = i + 5;
    float jx = j + 5;

    float xs = round(x + ( -jx*thScale*si + ix*thScale*co));
    float ys = round(y + ( jx*thScale*co + ix*thScale*si));
    
    // There are 81 work items per work group, logically
    // arranged into 9x9
    int k = i + (tha / 9);
    int l = j + (tha % 9);

    // Get coords of sample point on the rotated axis
    int sample_x = round(x + (-l*thScale*si + k*thScale*co));
    int sample_y = round(y + ( l*thScale*co + k*thScale*si));

    // Get the gaussian weighted x and y responses
    float gauss_s1 = gaussian((float)(xs-sample_x), (float)(ys-sample_y), 
                              2.5f*thScale);
    float rx = haarX(intImage, width, height, sample_y, sample_x, 
                     2*round(thScale));
    float ry = haarY(intImage, width, height, sample_y, sample_x, 
                     2*round(thScale));

    //Get the gaussian weighted x and y responses on rotated axis
    float rrx = gauss_s1*(-rx*si + ry*co);
    float rry = gauss_s1*(rx*co + ry*si);
    desc[tha].x = rrx;
    desc[tha].y = rry;
    desc[tha].z = fabs(rrx);
    desc[tha].w = fabs(rry);
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // Call summer function (result goes in index 0)
    sumDesc(desc, tha, DES_THREADS);
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if(tha == 0) 
    {
        // There are 16 work groups per i-point
        int dpos= bIdy * 16 + bIdx;
        float gauss_s2 = gaussian(cx-2.0f,cy-2.0f,1.5f);
        
        desc[0] *= gauss_s2;
        
        // Store the descriptor
        surfDescriptor[dpos] = desc[0];

        // Store the descriptor length
        descLength[bIdy * get_num_groups(0) + bIdx] = dot(desc[0], desc[0]);	
    }  

}
