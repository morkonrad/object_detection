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

#ifdef IMAGES_SUPPORTED
// CLK_ADDRESS_CLAMP returns (0,0,0,1) for out of bounds accesses
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP           |
                               CLK_FILTER_NEAREST;
#endif

#ifdef IMAGES_SUPPORTED
typedef __read_only image2d_t int_img_t;
#else
typedef __global float* int_img_t;
#endif

#ifdef IMAGES_SUPPORTED
typedef __write_only image2d_t det_layer_t; 
#else
typedef __global float* det_layer_t;
#endif

#ifdef IMAGES_SUPPORTED
typedef __write_only image2d_t lap_t; 
#else
typedef __global int* lap_t;
#endif

float 
BoxIntegral(int_img_t data, int width, int height, 
            int row, int col, int rows, int cols) 
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

// Compute the hessian determinant 
__kernel void 
hessian_det(
    int_img_t img,              // integral image
    int width,                  // integral image width
    int height,                 // integral image height
    det_layer_t responses,      // hessian determinant 
    lap_t laplacians,           // laplacian values 
    int layerWidth,             
    int layerHeight,
    int step,                   // determinant step size 
    int filter)                 // determinant filter size 
{

    int l, w, b;
    float Dxx, Dyy, Dxy, inverse_area;

    int idx = get_global_id(0);
    int idy = get_global_id(1);

    w = filter;                  // filter size
    l = filter/3;                // lobe for this filter              
    b = (filter - 1)/ 2 + 1;     // border for this filter   
    inverse_area = 1.0f/(w * w); // normalization factor

    int r = idy * step;
    int c = idx * step;

    // Have threads accessing out-of-bounds data return immediately
    if(r >= height || c >= width) {
        return;
    }
        
    Dxx = BoxIntegral(img, width, height, r - l + 1, c - b, 2*l - 1, w) -
          BoxIntegral(img, width, height, r - l + 1, c - l / 2, 2*l - 1, l)*3;

    Dyy = BoxIntegral(img, width, height, r - b, c - l + 1, w, 2*l - 1) -
          BoxIntegral(img, width, height, r - l / 2, c - l + 1, l, 2*l - 1)*3;

    Dxy = BoxIntegral(img, width, height, r - l, c + 1, l, l) +
          BoxIntegral(img, width, height, r + 1, c - l, l, l) -
          BoxIntegral(img, width, height, r - l, c - l, l, l) -
          BoxIntegral(img, width, height, r + 1, c + 1, l, l);

    // Normalize the filter responses with respect to their size
    Dxx *= inverse_area;
    Dyy *= inverse_area;
    Dxy *= inverse_area;

    // Save the determinant of hessian response
#ifdef IMAGES_SUPPORTED
    float4 determinant = {0.0f, 0.0f, 0.0f, 0.0f};
    determinant.x = (Dxx*Dyy - 0.81f*Dxy*Dxy);
    
    int4 laplacian = {0, 0, 0, 0};
    laplacian.x = (Dxx + Dyy >= 0 ? 1 : 0);
    
    write_imagef(responses, (int2)(idx,idy), determinant);
    write_imagei(laplacians, (int2)(idx,idy), laplacian);
#else

    float determinant = (Dxx*Dyy - 0.81f*Dxy*Dxy);
    int laplacian = (Dxx + Dyy >= 0 ? 1 : 0);
    
    responses[idy*layerWidth+idx] = determinant;
    laplacians[idy*layerWidth+idx] = laplacian;
#endif
}



