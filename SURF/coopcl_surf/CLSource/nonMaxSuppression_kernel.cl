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

/**
 *  This kernel performs non-max suppression.  Each point and it's surrounding area
 *  are investigated.  If a point is found to be the local maximum, it is returned,
 *  otherwise it is suppressed.
 **/
 
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable 
 
#ifdef IMAGES_SUPPORTED
// CLK_ADDRESS_CLAMP returns (0,0,0,1) for out of bounds accesses
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP           |  
                               CLK_FILTER_NEAREST;
#endif

#ifdef IMAGES_SUPPORTED
typedef __read_only image2d_t det_layer_t; 
#else
typedef __global float* det_layer_t;
#endif

#ifdef IMAGES_SUPPORTED
typedef __read_only image2d_t lap_t; 
#else
typedef __global int* lap_t;
#endif

int getLaplacian(lap_t layer, int c, int r, int width)
{
    int laplacian; 
    
#ifdef IMAGES_SUPPORTED
    laplacian = read_imagei(layer, sampler, (int2)(c,r)).x;
#else 
    laplacian = layer[r*width+c];   
#endif

    return laplacian;
}


float getResponse(det_layer_t layer, int c, int r, int width, int scale)
{ 
    float val;
    
#ifdef IMAGES_SUPPORTED
    val = read_imagef(layer, sampler, (int2)(c*scale,r*scale)).x;
#else
    int row = r*scale;
    val = layer[r*scale*width+c*scale];
#endif

    return val;
}


bool interpolateExtremum(      int  r, 
                               int  c, 					
                   __global    int* ipt_count,  
                            float2* pos,
                             float* det_scale,
                               int* laplacian,
                       det_layer_t  t,
                               int  tWidth,
                               int  tHeight,
                               int  tStep,
                       det_layer_t  m,
                             lap_t  mLaplacian,
                               int  mWidth,
                               int  mHeight,
                               int  mFilter,  
                       det_layer_t  b,
                               int  bWidth,
                               int  bHeight,
                               int  bFilter)
{

    // ---------------------------------------
    // Step 1: Calculate the 3D derivative
    // ---------------------------------------
    int mScale = mWidth/tWidth;

    int bScale = bWidth/tWidth;
    
    float dx, dy, ds; 

    dx = (getResponse(m, c+1, r,   mWidth, mScale) - getResponse(m, c-1, r,   mWidth, mScale)) / 2.0f;
    dy = (getResponse(m, c,   r+1, mWidth, mScale) - getResponse(m, c,   r-1, mWidth, mScale)) / 2.0f;
    ds = (getResponse(t, c,   r,   tWidth, 1)      - getResponse(b, c,   r,   bWidth, bScale)) / 2.0f;
    
    // ---------------------------------------
    // Step 2: Calculate the inverse Hessian
    // ---------------------------------------
    
    float v;
    
    float dxx, dyy, dss, dxy, dxs, dys;

    v = getResponse(m, c, r, mWidth, mScale);

    dxx =  getResponse(m, c+1, r,   mWidth, mScale) + getResponse(m, c-1, r,   mWidth, mScale) - 2.0f*v;
    dyy =  getResponse(m, c,   r+1, mWidth, mScale) + getResponse(m, c,   r-1, mWidth, mScale) - 2.0f*v;
    dss =  getResponse(t, c,   r,   tWidth, 1)      + getResponse(b, c,   r,   bWidth, bScale) - 2.0f*v;
    dxy = (getResponse(m, c+1, r+1, mWidth, mScale) - getResponse(m, c-1, r+1, mWidth, mScale) -
           getResponse(m, c+1, r-1, mWidth, mScale) + getResponse(m, c-1, r-1, mWidth, mScale))/4.0f;
    dxs = (getResponse(t, c+1, r,   tWidth, 1)      - getResponse(t, c-1, r,   tWidth, 1) -
           getResponse(b, c+1, r,   bWidth, bScale) + getResponse(b, c-1, r,   bWidth, bScale))/4.0f;
    dys = (getResponse(t, c,   r+1, tWidth, 1)      - getResponse(t, c,   r-1, tWidth, 1) -
           getResponse(b, c,   r+1, bWidth, bScale) + getResponse(b, c,   r-1, bWidth, bScale))/4.0f;

    float H0 = dxx;
    float H1 = dxy;
    float H2 = dxs;
    float H3 = dxy;
    float H4 = dyy;
    float H5 = dys;
    float H6 = dxs;
    float H7 = dys;
    float H8 = dss;

    // NOTE Although the inputs are the same, the value of determinant (and
    //      therefore invdet) vary from the CPU version
    
    float determinant =   
         H0*(H4*H8-H7*H5) -
         H1*(H3*H8-H5*H6) +
         H2*(H3*H7-H4*H6);
         
    float invdet = 1.0f / determinant;
       
    float invH0 =  (H4*H8-H7*H5)*invdet;
    float invH1 = -(H3*H8-H5*H6)*invdet;
    float invH2 =  (H3*H7-H6*H4)*invdet;
    float invH3 = -(H1*H8-H2*H7)*invdet;
    float invH4 =  (H0*H8-H2*H6)*invdet;
    float invH5 = -(H0*H7-H6*H1)*invdet;
    float invH6 =  (H1*H5-H2*H4)*invdet;
    float invH7 = -(H0*H5-H3*H2)*invdet;
    float invH8 =  (H0*H4-H3*H1)*invdet;
    
    // ---------------------------------------
    // Step 3: Multiply derivative and Hessian
    // ---------------------------------------
    
    float xi = 0.0f, xr = 0.0f, xc = 0.0f;
    
    xc =  invH0 * dx * -1.0f;
    xc += invH1 * dy * -1.0f;
    xc += invH2 * ds * -1.0f;
    
    xr =  invH3 * dx * -1.0f;
    xr += invH4 * dy * -1.0f;
    xr += invH5 * ds * -1.0f;
    
    xi =  invH6 * dx * -1.0f;
    xi += invH7 * dy * -1.0f;
    xi += invH8 * ds * -1.0f;

    // Check if point is sufficiently close to the actual extremum
    if(fabs(xi) < 0.5f && fabs(xr) < 0.5f && fabs(xc) < 0.5f)
    {
        int filterStep = mFilter - bFilter;
        
        // Store values related to interest point
        (*pos).x = (float)((c + xc)*tStep);    
        (*pos).y = (float)((r + xr)*tStep);    
        *det_scale = (float)(0.1333f)*(mFilter + (xi*filterStep));   
        
        int scale = mWidth/tWidth;  
        *laplacian = getLaplacian(mLaplacian, c*scale, r*scale, mWidth); 
       
        return true;
    }
    return false;
}


//! Check whether point really is a maximum
bool isExtremum( det_layer_t  t,
                         int  tWidth,
                         int  tHeight,
                         int  tFilter,
                         int  tStep,
                 det_layer_t  m,
                         int  mWidth,
                         int  mHeight,  
                 det_layer_t  b,
                         int  bWidth,
                         int  bHeight,                
                         int  c, 
                         int  r,
                       float  threshold)
{


    int layerBorder = (tFilter+1)/(2*tStep);

    // If any accesses would read out-of-bounds, this point is not a maximum
    if(r <= layerBorder || r >= tHeight - layerBorder ||
       c <= layerBorder || c >= tWidth - layerBorder) {
       
       return false;
    }
   
    int mScale = mWidth/tWidth;
    
    // Candidate for local maximum
    float candidate = getResponse(m, c, r, mWidth, mScale);
    
    if(candidate < threshold) {
        return false;
    }
    
    // If any response in 3x3x3 is greater candidate not maximum
    float localMax =          getResponse(t, c-1, r-1, tWidth, 1);
    localMax = fmax(localMax, getResponse(t, c,   r-1, tWidth, 1));
    localMax = fmax(localMax, getResponse(t, c+1, r-1, tWidth, 1));
    localMax = fmax(localMax, getResponse(t, c-1, r,   tWidth, 1));
    localMax = fmax(localMax, getResponse(t, c,   r,   tWidth, 1));
    localMax = fmax(localMax, getResponse(t, c+1, r,   tWidth, 1));
    localMax = fmax(localMax, getResponse(t, c-1, r+1, tWidth, 1));
    localMax = fmax(localMax, getResponse(t, c,   r+1, tWidth, 1));
    localMax = fmax(localMax, getResponse(t, c+1, r+1, tWidth, 1));
    
    int bScale = bWidth/tWidth;

    localMax = fmax(localMax, getResponse(b, c-1, r-1, bWidth, bScale));
    localMax = fmax(localMax, getResponse(b, c,   r-1, bWidth, bScale));
    localMax = fmax(localMax, getResponse(b, c+1, r-1, bWidth, bScale));
    localMax = fmax(localMax, getResponse(b, c-1, r,   bWidth, bScale));
    localMax = fmax(localMax, getResponse(b, c,   r,   bWidth, bScale));
    localMax = fmax(localMax, getResponse(b, c+1, r,   bWidth, bScale));
    localMax = fmax(localMax, getResponse(b, c-1, r+1, bWidth, bScale));
    localMax = fmax(localMax, getResponse(b, c,   r+1, bWidth, bScale));
    localMax = fmax(localMax, getResponse(b, c+1, r+1, bWidth, bScale));
    
    //int mScale = mWidth/tWidth;

    localMax = fmax(localMax, getResponse(m, c-1, r-1, mWidth, mScale));
    localMax = fmax(localMax, getResponse(m, c,   r-1, mWidth, mScale));
    localMax = fmax(localMax, getResponse(m, c+1, r-1, mWidth, mScale));
    localMax = fmax(localMax, getResponse(m, c-1, r,   mWidth, mScale));
    // This is the candidate pixel
    localMax = fmax(localMax, getResponse(m, c+1, r,   mWidth, mScale));
    localMax = fmax(localMax, getResponse(m, c-1, r+1, mWidth, mScale));
    localMax = fmax(localMax, getResponse(m, c,   r+1, mWidth, mScale));
    localMax = fmax(localMax, getResponse(m, c+1, r+1, mWidth, mScale));
    
    // If localMax > candidate, candidate is not the local maxima
    if(localMax > candidate) {
       return false;
    }
    
    return true;   
}


/* 
 * We don't know how many Ipoints there will be until after this kernel,
 * but we have to have enough space allocated beforehand.  Before a work item
 * writes to a location, he should make sure that there is space available.
 * After the kernel we'll check to make sure ipt_count is less than the 
 * allocated space, and if not, we'll run it again.
 */
__kernel
void non_max_supression(
                 det_layer_t  tResponse,
                         int  tWidth,
                         int  tHeight,
                         int  tFilter,
                         int  tStep,
                 det_layer_t  mResponse,
                       lap_t  mLaplacian,
                         int  mWidth,
                         int  mHeight,  
                         int  mFilter,
                 det_layer_t  bResponse,
                         int  bWidth,
                         int  bHeight,
                         int  bFilter,  
             __global    int* ipt_count,			
             __global float2* d_pixPos,
             __global  float* d_scale,
             __global    int* d_laplacian,   
                         int  maxPoints,
                       float  threshold)
{
    
    int r = get_global_id(1);
    int c = get_global_id(0);            
    
    float2 pixpos;
    float scale;
    int laplacian;
    
    // Check the block extremum is an extremum across boundaries.          
    if(isExtremum(tResponse, tWidth, tHeight, tFilter, tStep, mResponse,
                  mWidth, mHeight, bResponse, bWidth, bHeight, c, r, threshold))
    {  
        if(interpolateExtremum(r, c, ipt_count, &pixpos, &scale, 
            &laplacian, tResponse, tWidth, tHeight, tStep, mResponse, 
            mLaplacian, mWidth, mHeight, mFilter, bResponse, bWidth, bHeight, 
            bFilter)) {      			 			

            int index = atom_add(&ipt_count[0], 1);
            if(index < maxPoints) {
                d_pixPos[index] = pixpos;
                d_scale[index] = scale;
                d_laplacian[index] = laplacian;
            }
        }
    }
}