#include <cassert>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "clDriver.h"
#include "sift.h"

namespace SIFT_tasks 
{	

#define _SIFT_SIGMA 1.6f
#define _SIFT_CONTR_THR 0.065f
#define _SIFT_INTVLS  3
#define _SIFT_MAX_INTERP_STEPS 5
#define _SIFT_CURV_THR 10

#define _SIFT_CONTR_THR_1  (0.5 * (_SIFT_CONTR_THR/_SIFT_INTVLS))
#define _SIFT_IMG_BORDER  5
#define _SIFT_MAX_SCALES (_SIFT_INTVLS+3)
#define _SIFT_MAX_OCTAVES 3

#define _SIFT_ORI_HIST_BINS  36
#define _SIFT_ORIENT_GPU_BLOCKSIZE 16
#define _ORIENT_INIT_CONST (-1.0f)
#define _SIFT_ORI_SIG_FCTR  1.5f
#define _SIFT_ORI_RADIUS  (3 * _SIFT_ORI_SIG_FCTR) 
#define _SIFT_ORI_PEAK_RATIO  0.8f

#define _SIFT_DESCR_HIST_CELLS 16
#define _SIFT_DESCR_HIST_BINS 8
#define _SIFT_DESC_GPU_BLOCKSIZE (_SIFT_DESCR_HIST_CELLS*_SIFT_DESCR_HIST_BINS)

#define _SIFT_DESCR_SCL_FCTR  3.f
#define _SIFT_DESCR_MAG_THR  0.2f
#define _SIFT_INT_DESCR_FCTR  512.f

constexpr auto task_defines = R"(

#define MIN(a,b)    ((a)>(b)?(b):(a))
#define MAXV(A, B) 	((A>=B)?(A):(B))
#define MINV(A, B) 	((A<=B)?(A):(B))
#define ABS(A) 		(((A)>=0)?(A):(-A))
#ifndef M_PI
#define M_PI		3.14159265358979323846
#endif
#define M_PI2       (2.0F * M_PI)
#define _RPI (4.0/ M_PI)	// bins/rad
)";

constexpr auto task_Color = R"(
__kernel
void kColor_interleaved_ch( const global uchar* restrict inputImage,
             global float* restrict outputImage,const int w)
{
    const int x = get_global_id(0);	
    const int y = get_global_id(1);   		
    const int pix = mad24(y,w,x);
	
    const float3 coefBGR = {0.114,0.587,0.299};
    const float val_scale = 0.0039215686; // 1/255
	const uchar3 vpix = vload3(pix,inputImage);  
	
	float3 fpix;
    fpix.x = (float)vpix.x;
    fpix.y = (float)vpix.y;
    fpix.z = (float)vpix.z;  			
	outputImage[pix] = (dot(fpix,coefBGR))*val_scale;			
}

__kernel
void kColor( const global uchar* restrict inputImage,
             global float* restrict outputImage, const int w)
{
    const int x = get_global_id(0);	
    const int y = get_global_id(1);   		

    const int pix_R = mad24(y,w,x);
	const int pix_G = mad24(y,w,x)+w;
	const int pix_B = mad24(y,w,x)+2*w;
	
    const float3 coefBGR = {0.114,0.587,0.299};
    const float val_scale = 0.0039215686; // 1/255
	const uchar3 vpix = {inputImage[pix_B],inputImage[pix_G],inputImage[pix_R]};
	float3 fpix;
    fpix.x = (float)vpix.x;
    fpix.y = (float)vpix.y;
    fpix.z = (float)vpix.z;  			
	outputImage[pix_R] = dot(fpix,coefBGR)*val_scale;	
}
)";
constexpr auto task_Down = R"(
__kernel
void kDown( global float* restrict dst,
            const global float* restrict src,
            const int wo,
            const int d,
			const int w,const int h)
{	
    const int x = get_global_id(0);
    const int y = get_global_id(1);    

    if (x < w && y < h)
    {
        const int pix_dst = mad24(y,w,x);
        const int pix_src = mad24(y*d,wo,x*d);
        dst[pix_dst] = src[pix_src];
    }
}
__kernel
void kDown_pyramid( global float* restrict dst,
            const global float* restrict src,
            const int wo,const int d,
			const int w,const int h,
			const int ss_img_id)
{	
    const int x = get_global_id(0);
    const int y = get_global_id(1);    

    if (x < w && y < h)
    {
        const int pix_dst = mad24(y,w,x);
		//src is a pyramid(stack) with images. Read the second one ( with idx=1, shift w*h)
		//d is scaling factor, wo is non-scaled img width, w,h are scaled width,height 
		const int offset_ss_image = ss_img_id*w*d*h*d;
        const int pix_src = mad24(y*d,wo,x*d)+offset_ss_image; 
        dst[pix_dst] = src[pix_src];
    }
}
)";
constexpr auto task_Resize = R"(
__kernel
void kResize_NN_1ch_32b( global float* restrict dst,
						const global float* restrict src,
						const int width_dst,const int height_dst,            
						const int width_src,const int height_src,
						const float ratio_x,const float ratio_y)
{	
    const int x = get_global_id(0);
    const int y = get_global_id(1);    
	
	if (x > width_dst-1 || y > height_dst-1) return;

	const float fy = ((float)y+0.4995f)/ratio_y;
	const float fx = ((float)x+0.4995f)/ratio_x;	
	
	const int pix_dst = mad24(y,width_dst,x);
    const int pix_src = mad24(fy,width_src,fx);

    if ( fx < width_src && fy < height_src)    
        dst[pix_dst] = src[pix_src];        		
    else	
		dst[pix_dst] = 0;
}

__kernel
void kResize_NN_1ch_8b( global uchar* restrict dst,
						const global uchar* restrict src,
						const int width_dst,const int height_dst,            
						const int width_src,const int height_src,
						const float ratio_x,const float ratio_y)
{	
    const int x = get_global_id(0);
    const int y = get_global_id(1);    
	
	if ( x > width_dst-1 || y > height_dst-1) return;

	const float fy = ((float)y+0.4995f)/ratio_y;
	const float fx = ((float)x+0.4995f)/ratio_x;	
	
	const int pix_dst = mad24(y,width_dst,x);
    const int pix_src = mad24(fy,width_src,fx);

    if ( fx < width_src && fy < height_src)    
        dst[pix_dst] = src[pix_src];        		
    else	
		dst[pix_dst] = 0;
}

__kernel
void kResize_NN_3ch_8b( global uchar* restrict dst,
						const global uchar* restrict src,
						const int width_dst,const int height_dst,            
						const int width_src,const int height_src,
						const float ratio_x,const float ratio_y)
{	
    const int x = get_global_id(0);
    const int y = get_global_id(1);    
	
	if ( x > width_dst-1 || y > height_dst-1) return;

	const float fy = ((float)y+0.4995f)/ratio_y;
	const float fx = ((float)x+0.4995f)/ratio_x;	
	
	const int pix_dst = mad24(y,width_dst,x);
    const int pix_src = mad24(fy,width_src,fx);

    if ( fx < width_src && fy < height_src)
	{    
		const uchar3 val = vload3(pix_src,src);	
        vstore3(val,pix_dst,dst);
	}
    else	
		vstore3(0,pix_dst,dst);
}

)";
	constexpr auto task_Blur = R"(
__kernel
void kBlurV(const global float* restrict inputImage,			
			const global float* restrict filter,   
			global float* restrict outputImage,      
			const int w,const int h,const int fr,const int ssid)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);  	
    float prod = 0.0f;		
	const int offset_ss_img = ssid<0?0:ssid*w*h;
	for( int i=-fr;i<fr;i++){	
		const float pixv = y+i>h-1?0:y+i<0?0:inputImage[(y+i)*w+x];
		prod += filter[fr+i]*pixv;
	}
    outputImage[y*w+x+offset_ss_img] = prod;		
}
__kernel
void kBlurH( const global float* restrict inputImage,
			const global float* restrict filter,
            global float* restrict outputImage,
			const int w,const int h,const int fr,const int ssid)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);    
	float prod = 0.0f;			
	const int offset_ss_img = ssid<0?0:ssid*w*h; 
	for( int i=-fr;i<fr;i++){	
		const float pixv = x+i>w-1?0:x+i<0?0:inputImage[y*w+(x+i)+offset_ss_img];
		prod += filter[fr+i]*pixv;
	}
    outputImage[y*w+x] = prod;		
}

)";

	constexpr auto task_Diff = R"(
__kernel
void kDiff( const global float* restrict inputImage_scale_space,			
            global float* restrict outputImage,
			const int w,const int h, const int ssid)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);    	

    const int pix = mad24(y,w,x);
	const int pix_prev = pix+ssid*w*h;
	const int pix_next = pix+(ssid+1)*w*h;

    outputImage[pix] = (inputImage_scale_space[pix_next]-inputImage_scale_space[pix_prev]);	
}
)";

	constexpr auto task_Detector = R"(
__kernel
void kDetector( 
global float4* restrict features,
const global float* restrict dog_p,
const global float* restrict dog_c,
const global float* restrict dog_n,
global uint* restrict count,
const int w,const int h,
const int s,const int o,
const uint max_feat)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	float curr_pix;

	if(y <= _SIFT_IMG_BORDER)return;
	if(y >= h-_SIFT_IMG_BORDER)return;
	if(x <= _SIFT_IMG_BORDER)return;
	if(x >= w - _SIFT_IMG_BORDER)return;

	curr_pix = dog_c[mad24(y,w,x)];

	if(fabs(curr_pix) <= _SIFT_CONTR_THR_1)return;

	#define at(pt, y, x) (*((pt) + (y)*w + (x)))

	#define LOC_EXTREMUM(PXL, IMG, CMP, SGN)			\
	( PXL CMP at(IMG ## _p, y, x-1)      &&		\
	PXL CMP at(IMG ## _p, y, x)    &&		\
	PXL CMP at(IMG ## _p, y, x+1)    &&		\
	PXL CMP at(IMG ## _p, y+1, x-1)    &&		\
	PXL CMP at(IMG ## _p, y+1, x)  &&		\
	PXL CMP at(IMG ## _p, y+1, x+1)  &&		\
	PXL CMP at(IMG ## _p, y-1, x-1)    &&		\
	PXL CMP at(IMG ## _p, y-1, x)  &&		\
	PXL CMP at(IMG ## _p, y-1, x-1)  &&		\
	\
	PXL CMP at(IMG ## _c, y, x-1)   &&		\
	PXL CMP at(IMG ## _c, y, x+1)   &&		\
	PXL CMP at(IMG ## _c, y+1, x-1)   &&		\
	PXL CMP at(IMG ## _c, y+1, x) &&		\
	PXL CMP at(IMG ## _c, y+1, x+1) &&		\
	PXL CMP at(IMG ## _c, y-1, x-1)   &&		\
	PXL CMP at(IMG ## _c, y-1, x) &&		\
	PXL CMP at(IMG ## _c, y-1, x+1) &&		\
	\
	PXL CMP at(IMG ## _n, y, x-1)      &&		\
	PXL CMP at(IMG ## _n, y, x)    &&		\
	PXL CMP at(IMG ## _n, y, x+1)    &&		\
	PXL CMP at(IMG ## _n, y+1, x-1)    &&		\
	PXL CMP at(IMG ## _n, y+1, x)  &&		\
	PXL CMP at(IMG ## _n, y+1, x+1)  &&		\
	PXL CMP at(IMG ## _n, y-1, x-1)    &&		\
	PXL CMP at(IMG ## _n, y-1, x)  &&		\
	PXL CMP at(IMG ## _n, y-1, x+1))

	if (LOC_EXTREMUM(curr_pix,dog,>,+) || LOC_EXTREMUM(curr_pix,dog,<,-))
	{
		float Dxx,Dyy,Dss,Dxy,Dxs,Dys,trH,detH;

		//float3 Hrow0,Hrow1,Hrow2,grad;
		float3 grad;

		float3 offset3 = {0.0f,0.0f,0.0f};

		int ty,tx,ts,iter;

		ty = y;//row transition
		tx = x;//col transition
		ts = s;//scale transition

		for(iter=0;iter<_SIFT_MAX_INTERP_STEPS;iter++)
		{
			// compute gradient
			grad.x = 0.5f*(dog_c[mad24(ty,w,tx+1)]  - dog_c[mad24(ty,w,tx-1)]);
			grad.y = 0.5f*(dog_c[mad24(ty+1,w,tx)] - dog_c[mad24(ty-1,w,tx)]);
			grad.z = 0.5f*(dog_n[mad24(ty,w,tx)] - dog_p[mad24(ty,w,tx)]);

			// compute Hessian
			Dxx = dog_c[mad24(ty,w,tx+1)] - 2.0f * dog_c[mad24(ty,w,tx)] + dog_c[mad24(ty,w,tx-1)];
			Dyy = dog_c[mad24(ty+1,w,tx)]- 2.0f * dog_c[mad24(ty,w,tx)] + dog_c[mad24(ty-1,w,tx)];
			Dss = dog_n[mad24(ty,w,tx)] - 2.0f * dog_c[mad24(ty,w,tx)] + dog_p[mad24(ty,w,tx)];

			// compute partial derivative
			Dxy = 0.25f * (dog_c[mad24(ty+1,w,tx+1)] + dog_c[mad24(ty-1,w, tx-1)]	- dog_c[mad24(ty+1,w,tx-1)] - dog_c[mad24(ty-1,w,tx+1)]);
			Dxs = 0.25f * (dog_n[mad24(ty,w,tx+1)]	+ dog_p[mad24(ty,w,tx-1)]		- dog_n[mad24(ty,w,tx-1)]	- dog_p[mad24(ty,w,tx+1)]);
			Dys = 0.25f * (dog_n[mad24(ty+1,w,tx)]	+ dog_p[mad24( ty-1,w,tx)]		- dog_p[mad24(ty+1,w,tx)]	- dog_n[mad24(ty-1,w,tx)]);

			float3 Hrow0,Hrow1,Hrow2;
		
			Hrow0.x = Dyy*Dss - Dys*Dys;//A		
			Hrow1.x = Dys*Dxs - Dxy*Dss;//B		
			Hrow2.x = Dxy*Dys - Dyy*Dxs;//C	
			detH = Dxx*Hrow0.x+Dxy*Hrow1.x+Dxs*Hrow2.x;
			
			Hrow0.y = Dxs*Dys - Dxy*Dss;//D		
			Hrow0.z = Dxy*Dys - Dxs*Dyy;//G		
			Hrow1.y = Dxx*Dss - Dxs*Dxs;//E		
			Hrow1.z = Dxs*Dxy - Dxx*Dys;//H		
			Hrow2.y = Dxy*Dxs - Dxx*Dys;//F		
			Hrow2.z = Dxx*Dyy - Dxy*Dxy;//I
			
			if(detH == 0.0f)return; // non invertible !
			float dv = native_divide(-1.0f,detH);

			Hrow0*= dv;
			Hrow1*= dv;
			Hrow2*= dv;

			offset3.x = dot(Hrow0,grad);
			offset3.y = dot(Hrow1,grad);
			offset3.z = dot(Hrow2,grad);
		
			if(fabs(offset3.x)<0.5f && fabs(offset3.y)<0.5f && fabs(offset3.z)<0.5f)break;
		
			tx += round(offset3.x);
			if(tx<_SIFT_IMG_BORDER || tx>=w-_SIFT_IMG_BORDER)return;

			ty += round(offset3.y);
			if(ty<_SIFT_IMG_BORDER || ty>=h-_SIFT_IMG_BORDER)return;

			ts += round(offset3.z);
			if(ts<1 || ts>_SIFT_INTVLS)return;
		}

		if(iter>=_SIFT_MAX_INTERP_STEPS)return; // if extremum not sub localized then reject this sample !		
		
		trH  = Dxx + Dyy;
		detH = Dxx * Dyy - Dxy * Dxy;

		//if(detH <= 0.0f)return;
		if(fabs(detH) <= 1e-8)return;
		if( (trH*trH)/detH >= ((_SIFT_CURV_THR+1)*(_SIFT_CURV_THR+1)/_SIFT_CURV_THR))return;		
		
		curr_pix = ts==1?dog_p[mad24(ty,w,tx)]:ts==2?dog_c[mad24(ty,w,tx)]:dog_n[mad24(ty,w,tx)];

		// reuse detH as contrast value
		detH = curr_pix + 0.5f * (grad.x * offset3.x + grad.y * offset3.y + grad.z * offset3.z);

		if( fabs(detH) > (float)(_SIFT_CONTR_THR/_SIFT_INTVLS))
		{
			uint fid = atomic_inc(count);
			if(fid<max_feat)
			{
				features[fid].x = (tx+offset3.x)*native_powr(2.0f,(float)o);
				features[fid].y = (ty+offset3.y)*native_powr(2.0f,(float)o);;
				features[fid].z = _SIFT_SIGMA * native_powr(2.0f, (((float)ts+offset3.z)/(float)_SIFT_INTVLS));  //sigma 
				features[fid].w = ts;																					//scl_id
			}
		}
	}
}
)";

constexpr auto task_Reset = R"(
__kernel 
void kReset(global float4* restrict features, global float* restrict feature_orientations)
{
	const int tid = get_global_id(0);
	features[tid] = -1.0f;
	feature_orientations[tid] = 0;
}
)";

constexpr auto task_Orientation = R"(
__kernel
void kOrientation(const global float4* restrict features,
                  const global float* restrict GaussPyramid,
				  const int w, const int h,const int found_fetures,const oid,
				  global float* restrict feature_orientations)
{
    const int lid = get_local_id(0);
    //const int bid = get_group_id(0);   
	const int bid = (get_global_id(0)-lid)/get_local_size(0);   

	if(bid>found_fetures-1)return; //avoid extra threads/groups
	if( features[bid].w<0 || features[bid].w > _SIFT_INTVLS)return;    	

    // shared memory - sub histograms
    local float hist_sub[_SIFT_ORIENT_GPU_BLOCKSIZE][_SIFT_ORI_HIST_BINS];

    #pragma unroll _SIFT_ORI_HIST_BINS
    for (int i = 0; i < _SIFT_ORI_HIST_BINS; i++)    
        hist_sub[lid][i] = 0.0f;        
    
    barrier(CLK_LOCAL_MEM_FENCE);

    const float exp_denom = features[bid].z*_SIFT_ORI_SIG_FCTR;
    const int rad  = (int)round(_SIFT_ORI_RADIUS*features[bid].z);
	
	const float exp_factor = -0.5f / (exp_denom * exp_denom);
    const int sq_thres  = rad * rad;

    const int offset_py = w*h*features[bid].w;

    const int x = features[bid].x / native_powr(2.0f,(float)oid); //rescale with octave_id
    const int y = features[bid].y / native_powr(2.0f,(float)oid);//rescale with octave_id

    const int xmin = MAXV(1, (x - rad));
    const int ymin = MAXV(1, (y - rad));
    const int xmax = MINV(w - 2, (x + rad));
    const int ymax = MINV(h - 2, (y + rad));
    const int wx = xmax - xmin + 1;
    const int hy = ymax - ymin + 1;
    const int loops = wx * hy;

    int bin = 0;	
    int i,j;

    #pragma unroll 4
    for(i = lid; i < loops; i+=_SIFT_ORIENT_GPU_BLOCKSIZE)
    {
        const int yy = i / wx + ymin;
        const int xx = i % wx + xmin;

        if( xx>0&&xx<w-1&&yy>0&&yy<h-1)
        {
            float dx = GaussPyramid[ mad24(yy,w,xx+1)+offset_py] - GaussPyramid[ mad24(yy,w,xx-1)+offset_py];//dx
            float dy = GaussPyramid[ mad24(yy-1,w,xx)+offset_py] - GaussPyramid[ mad24(yy+1,w,xx)+offset_py];//dy
			
            const float grad = native_sqrt(dx*dx+dy*dy);
            const float theta = atan2(dy,dx);
            
			dx = xx - x;
            dy = yy - y;

            const int sq_dist  = (int)(dx * dx + dy * dy);
            if (sq_dist > sq_thres) continue;
            
            bin = round( (_SIFT_ORI_HIST_BINS * (theta + M_PI)) / M_PI2); // angle in rad
            bin = (bin < _SIFT_ORI_HIST_BINS)?bin:0;

            hist_sub[lid][bin] += grad*native_exp(sq_dist * exp_factor);
        }		
    }

	barrier(CLK_LOCAL_MEM_FENCE);
	
    // reduction here
    // Now aggregate sub_histograms
    //------------------------------------	
	
	#pragma unroll _SIFT_ORI_HIST_BINS
    for (i = 0; i < _SIFT_ORI_HIST_BINS; i++)			
	{				
		if(lid<8){ hist_sub[lid][i] += hist_sub[lid+8][i]; }
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(lid<4){ hist_sub[lid][i] += hist_sub[lid+4][i]; }
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(lid<2){ hist_sub[lid][i] += hist_sub[lid+2][i]; }
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(lid==0){ hist_sub[0][i] += hist_sub[1][i]; }
		barrier(CLK_LOCAL_MEM_FENCE);
	}	
	barrier(CLK_LOCAL_MEM_FENCE);	

	float hp,hc,h0,hn,fbin,v;

    if(lid==0)
    {        
        // smooth histogram
        #pragma unroll
        for (j = 0; j < 2; j++)
        {            
			h0 = hist_sub[0][0];
            hp = hist_sub[0][_SIFT_ORI_HIST_BINS-1];

			#pragma unroll
            for( i = 0; i < _SIFT_ORI_HIST_BINS; i++ )
            {                
				hc = hist_sub[0][i];
                hist_sub[0][i] = 0.25 * hp + 0.5 * hist_sub[0][i] + 0.25 * ( ( i+1 == _SIFT_ORI_HIST_BINS )? h0 : hist_sub[0][i+1] );
                hp = hc;
            }
        }

        // find histogram maximum        
		v = hist_sub[0][0];

        #pragma unroll _SIFT_ORI_HIST_BINS
        for (i = 1; i< _SIFT_ORI_HIST_BINS; i++)
        {            
			if(hist_sub[0][i]>v)v = hist_sub[0][i];
        }

		feature_orientations[bid] = _ORIENT_INIT_CONST;

        // update maximum value
        v = _SIFT_ORI_PEAK_RATIO * v;

        // counter for orientation items
        //j = 0;

        //	find peaks, boundary of 80% of max
        #pragma unroll _SIFT_ORI_HIST_BINS
        for (i = 0; i < _SIFT_ORI_HIST_BINS; i++)
        {            
			hc = hist_sub[0][i];
            hp = hist_sub[0][(i==0)?(_SIFT_ORI_HIST_BINS-1):(i-1)];//l
			hn = hist_sub[0][((i+1)%_SIFT_ORI_HIST_BINS)];//r

            // find if a peak
            if (hc >= v && hc > hn && hc > hp)
            {
                // interpolate ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )
                fbin = (float)i + (0.5f *(hp - hn) / (hp-2.0*hc+hn));

                // clamp
                fbin = (fbin < 0.0) ? (fbin + (float)_SIFT_ORI_HIST_BINS) : ((fbin >= (float)_SIFT_ORI_HIST_BINS) ? (fbin - (float)_SIFT_ORI_HIST_BINS) : (fbin));
				feature_orientations[bid] = ((M_PI2 * fbin) / _SIFT_ORI_HIST_BINS) - M_PI;                
            }
        }
    }
})";

constexpr auto task_Descriptor = R"(
__kernel
void kDescriptor(	const global float4* restrict features,
					const global float* restrict orientations,
                    const global float* restrict GaussPyramid,
					const int w,const int h,const int found_fetures,const int oid,
					global float* restrict descriptors)
{
    // one feature/thread block
    //const int fid = get_group_id(0);	//feature id	 
	const int fid = (get_global_id(0)-get_local_id(0))/get_local_size(0);
   
	if( features[fid].w<0 || features[fid].w > _SIFT_INTVLS) return;
	if(fid>found_fetures-1) return; //avoid extra threads/groups

    // 16 cells/keypoint, 8 threads/cell, 9bin histogram/thread
    // 128 threads/keypoint (128 block size)
    __local float des[_SIFT_DESCR_HIST_CELLS][_SIFT_DESCR_HIST_BINS][9];

    const int lid = get_local_id(0) & 0x07; //0:7 id of threads inside cell
    const int bid = get_local_id(0) >> 3; //cell id 0:15

    // cell (x,y) ids-> 00,01,02,03,10,11,12,13,20,21,22,23,30,31,32,33
    const int ix = bid & 0x3;
    const int iy = bid >> 2;

    // initialize
    #pragma unroll
    for(int i=0;i<9;i++)
        des[bid][lid][i] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    const int x = features[fid].x/ native_powr(2.0f,(float)oid);//rescale with octave_id
    const int y = features[fid].y/ native_powr(2.0f,(float)oid);//rescale with octave_id
    
	const int offset_py = w*h*features[fid].w;

    const float ang  = -orientations[fid];	// reverse sign !!
    const float spt = fabs(features[fid].z * 3.0);

    const float s = native_sin(ang);
    const float c = native_cos(ang);

    const float cspt = c * spt;
    const float sspt = s * spt;

    const float crspt = c / spt;
    const float srspt = s / spt;

	const float offsetptx = ix - 1.5f;
    const float offsetpty = iy - 1.5f;

    const float ptx = cspt * offsetptx - sspt * offsetpty + x;
    const float pty = cspt * offsetpty + sspt * offsetptx + y;
    const float bsz =  fabs(cspt) + fabs(sspt);

    const int xmin = MAXV(1, floor(ptx - bsz) );
    const int ymin = MAXV(1, floor(pty - bsz) );
    const int xmax = MINV(w - 2, floor(ptx + bsz) );
    const int ymax = MINV(h - 2, floor(pty + bsz) );
    const int wx = xmax - xmin + 1;
    const int hy = ymax - ymin + 1;
    const int loops = wx * hy;

    for(int i = lid; i < loops; i+= 8)
    {
        const int yy = i / wx + ymin;
        const int xx = i % wx + xmin;

        float dx = xx - ptx;
        float dy = yy - pty;

        const float nx = crspt * dx + srspt * dy;
        const float ny = crspt * dy - srspt * dx;

        const float nxn = fabs(nx);
        const float nyn = fabs(ny);
		
        if(nxn < 1.0f && nyn < 1.0f)
        {	
            dx = GaussPyramid[mad24(yy,w,xx+1)+offset_py] - GaussPyramid[mad24(yy,w,xx-1)+offset_py];
            dy = GaussPyramid[mad24(yy+1,w,xx)+offset_py] - GaussPyramid[mad24(yy-1,w,xx)+offset_py];
			
            const float mod = native_sqrt(dx*dx+dy*dy);
            float th  = atan2(dy,dx);

            const float dnx = nx + offsetptx;
            const float dny = ny + offsetpty;

            const float ww = native_exp(-0.125f * (dnx * dnx + dny * dny));
            const float wx = 1.0 - nxn;
            const float wy = 1.0 - nyn;
            const float wgt = ww * wx * wy * mod;

            th -= ang;
            while (th < 0.0f) th += M_PI2;
            while (th >= M_PI2) th -= M_PI2;

            const float tth = th * _RPI;
            const int   fo0 = (int)floor(tth);
            const float do0 = tth - fo0;

            const float wgt1 = 1.0f - do0;
            const float wgt2 = do0;
            const int fo  = fo0 % _SIFT_DESCR_HIST_BINS;
			
            // 1D interpolate
            if(fo < _SIFT_DESCR_HIST_BINS)
            {
                des[bid][lid][fo] += (wgt1*wgt);
                des[bid][lid][fo+1] += (wgt2*wgt);
            }						
        }
    }
    des[bid][lid][0]+=des[bid][lid][8];	
    barrier(CLK_LOCAL_MEM_FENCE);

    float bin = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        bin += des[bid][i][lid];

    descriptors[fid*_SIFT_DESC_GPU_BLOCKSIZE+get_local_id(0)] = bin;	
}
)";

constexpr auto task_Normalize = R"(
__kernel
void kNormalize(global float* restrict descriptors)
{
    const int tid = get_global_id(0);

    float16 v0,v1;

    float tmp[128];

    int i;
    float norm1,norm2;

    for(i=0;i<128;i+=16)
    {
        v0 = vload16(0,&descriptors[tid*128+i]);
        v1 = v0*v0;//native_powr(v0,(float16)2.0);
        vstore16(v1,0,&tmp[i]);
    }

    norm1=0.0f;

    //reduce
    for(i=0;i<128;i++){
        norm1+=tmp[i]; }

    norm1 = native_rsqrt(norm1);

    for(i=0;i<128;i+=16)
    {
        v0 = vload16(0,&descriptors[tid*128+i]);
        v1 = v0*(float16)(norm1); // unit length
        v0 = clamp(v1,0.0,(float16)_SIFT_DESCR_MAG_THR); // clamp
        v1 = v0*v0;//native_powr(v0,(float16)2.0);
        vstore16(v0,0,&descriptors[tid*128+i]);
        vstore16(v1,0,&tmp[i]);
    }

    norm2 = 0.0f;

    //reduce
    for(i=0;i<128;i++){
        norm2+=tmp[i]; }

    norm2 = native_rsqrt(norm2);

    for(i=0;i<128;i+=16)
    {
        v0 = vload16(0,&descriptors[tid*128+i]);
        v1 = v0*(float16)(norm2); // unit length
        v0 = (float16)_SIFT_INT_DESCR_FCTR*v1;
        v1 = clamp(v0,0.0,(float16)255.0); // clamp
        vstore16(v1,0,&descriptors[tid*128+i]);
    }
}
)";

}

class sift_octave
{
	private:

	coopcl::clTask _task_Reset;
	coopcl::clTask _task_Color;	
	coopcl::clTask _task_Down_pyramid;
	coopcl::clTask _task_UpScale_1ch_32b;
	coopcl::clTask _task_UpScale_3ch_8b;

	std::unique_ptr<coopcl::clMemory> _imgColor_non_sampled{ nullptr };
	std::unique_ptr<coopcl::clMemory> _imgColor_resampled{ nullptr };
	std::unique_ptr<coopcl::clMemory> _imgGray_non_sampled{ nullptr };

	std::unique_ptr<coopcl::clMemory> _scale_space_pyramid;
	std::vector<std::unique_ptr<coopcl::clMemory>> _imgFilters;
	std::vector<std::unique_ptr<coopcl::clMemory>> _scale_images_intermidiate;		
	std::vector<std::unique_ptr<coopcl::clMemory>> _diff_scale_images;

	std::vector<std::unique_ptr<coopcl::clTask>> _task_BlurH;
	std::vector<std::unique_ptr<coopcl::clTask>> _task_BlurV;
	std::vector<std::unique_ptr<coopcl::clTask>> _task_Diff;
	
	std::vector<std::unique_ptr<coopcl::clTask>> _task_Detector;	
	std::unique_ptr<coopcl::clMemory> _counter_kp;
	std::unique_ptr<coopcl::clMemory> _kp_Detector{ nullptr };
	
	std::unique_ptr<coopcl::clTask> _task_Orientation{ nullptr };
	std::unique_ptr<coopcl::clMemory> _kp_Orientations{ nullptr };

	std::unique_ptr<coopcl::clTask> _task_Descriptor{ nullptr };
	std::unique_ptr<coopcl::clTask> _task_Normalize{ nullptr };
	std::unique_ptr<coopcl::clMemory> _kp_Descriptors{ nullptr };

	std::uint32_t _max_features{ 0 };
	size_t _octave_id{ 0 };
	size_t _scale_image_width{ 0 };
	size_t _scale_image_height{ 0 };
	size_t _input_image_channels{ 0 };
    bool _async_calls{ false };
    bool _up_scale{ false };
	
	const float _resample_ratio_y{ 2.0f };
	const float _resample_ratio_x{ 2.0f };

	std::vector<std::vector<float>> _scale_filters;
	std::vector<float> _filter_coef_sigmas;
	std::vector<std::uint32_t> _filter_coef_widths;

	int Build_separable_filter_cof(const float _sigma)
	{
		int status = 0;
		std::uint32_t s = 0;
		auto fcval = 0.f;
		auto sum = 0.f;
        auto sigma = _sigma <= 0 ? 1.6f : _sigma;
		_scale_filters.resize(_SIFT_MAX_SCALES);
		_filter_coef_sigmas.resize(_SIFT_MAX_SCALES);
		_filter_coef_widths.resize(_SIFT_MAX_SCALES);

		_filter_coef_sigmas[0] = sigma;
		float k = powf(2.0f, 1.0f / (_SIFT_INTVLS));// k=2^1/S

													// pre-compute Gaussian pyramid sigmas
													// This method calculates the Gaussian coefficients filters 
													// It calculates delta_sigmas = sqrt(sigma_next_lvl^2-sigma_prev_lvl^2)
													//----------------------------------------------------------------------------------
													// Implementation targets the semi-group property of Gauss_kernel
													//----------------------------------------------------------------------------------
		for (s = 1; s < _SIFT_MAX_SCALES; s++)
		{
			float lvl_sig_prev = sigma *powf(k, (float)(s - 1));
			float lvl_sig_next = lvl_sig_prev*k;
			auto delta_sig = (float)std::sqrt(lvl_sig_next*lvl_sig_next - lvl_sig_prev*lvl_sig_prev);
			_filter_coef_sigmas[s] = delta_sig;
		}

		//pre-compute Gaussian pyramid sigmas
		//This is non-incremental method !! (longer filter widths/radius)
		//sig0,ksig,k^2*sig,k^3*sig,k^4*sig,k^5*sig ...
		//----------------------------------------------------------------------------------
		// Implementation omits the semi-group property of Gauss_kernel
		//----------------------------------------------------------------------------------
		/*for (s = 0; s < MAX_SCALES; s++)
		_filter_coef_sigmas[s] = sigma*pow(k, s);*/

		size_t fc = 0;
		// compute Gauss kernels
		for (s = 0; s < _SIFT_MAX_SCALES; s++)
		{
			sum = 0.0f;

			auto& s_sigma = _filter_coef_sigmas[s];
			auto& filter_coef = _scale_filters[s];

			// Filter radius / discretized Gauss function with [-3.5sig:3,5sig]
			auto frad = (size_t)(std::ceil(3.5f * s_sigma));
			auto fil_width = 2 * frad + 1;

			_filter_coef_widths[s] = fil_width;
			filter_coef.resize(fil_width, 0.0f);

			for (fc = 0; fc < frad; fc++)
			{
				// Gauss filter is symmetric and separable
				filter_coef[fc] = filter_coef[2 * frad - fc] = fcval = float(std::exp(float(-0.5 *((fc - frad) * (fc - frad) / (_filter_coef_sigmas[s] * _filter_coef_sigmas[s])))));
				sum += 2 * fcval;
			}

			filter_coef[fc] = 1;
			sum += 1;

			// normalize
			fcval = 1.0f / sum;

			for (fc = 0; fc < 2 * frad + 1; fc++)
				filter_coef[fc] *= fcval;
		}
		return status;
	}
		
	int reset(coopcl::virtual_device& device,const float offload)
	{				
		auto ptr = (std::uint32_t*)_counter_kp->data();
		*ptr = 0;

		int err = device.execute_async(_task_Reset, offload, { _max_features,1,1 }, { 1,1,1 }, _kp_Detector, _kp_Orientations);
		if(err!=0)return err;
		return _task_Reset.wait();				
	}
	
	void copy_pixels_1ch8_to_1ch32(const std::uint8_t* input_image,const size_t width,const size_t height)
	{
		const auto items = width*height;
		if (items == 0)return;
		if (input_image == nullptr)return;
		//Copy single_channel_byte into 1_ch float
        float* ptr_pix = _up_scale?static_cast<float*>(_imgGray_non_sampled->data()):static_cast<float*>(_scale_space_pyramid->data());
		for (int i = 0; i < items; i++)
		{
			const auto fpix = static_cast<float>(input_image[i]) / 255.0f;
			ptr_pix[i] = fpix;
		}
	}

	std::string 
	store_task_in_file(const coopcl::clTask* task,
		const std::string& path,
		const size_t octave_id,
		const size_t scale_id)const
	{
		std::string file_name = path;
		file_name.append(task->name());
		file_name.append(std::to_string(octave_id));
		file_name.append(std::to_string(scale_id));

		std::string file_name_csv = file_name;
		file_name_csv.append("_obs.csv");
		std::ofstream ofs_observ_csv(file_name_csv);

#ifdef _TASK_INSTRUMENTATION_		
		task->write_task_profile(ofs_observ_csv);
#endif

		std::string file_name_off = file_name;
		file_name_off.append("_off.dat");

		std::string file_name_cpu = file_name;
		file_name_cpu.append("_cpu.dat");

		std::string file_name_gpu = file_name;
		file_name_gpu.append("_gpu.dat");

		std::ofstream ofs_off(file_name_off);
		std::ofstream ofs_cpu(file_name_cpu);
		std::ofstream ofs_gpu(file_name_gpu);

		std::stringstream obs_offload;
		std::stringstream obs_cpu;
		std::stringstream obs_gpu;
		task->write_records_to_stream(obs_offload,obs_cpu,obs_gpu);
		
		ofs_off << obs_offload.str();
		ofs_cpu << obs_cpu.str();
		ofs_gpu << obs_gpu.str();

		return file_name;
	}
	
	size_t found_features_cnt()const
	{
		if (_async_calls)
		{
			for (auto& td : _task_Detector)
				td->wait();
		}
		
		return _counter_kp->at<std::uint32_t>(0);
	}

	int call_sync_detector(const float offload,
        coopcl::virtual_device& device/*,
		std::unique_ptr<coopcl::clMemory>* scale_image_prev_octave = nullptr,
        coopcl::clTask* wait_task_prev_octave = nullptr*/)
	{

		int err = 0;
		err = reset(device, offload);
		if (err != 0)return err;

		const int i32w = _scale_image_width;
		const int i32h = _scale_image_height;

		const size_t gsx = i32w % 16 == 0 ? 16 : 1;
		const size_t gsy = i32h % 8 == 0 ? 8 : 1;

		if (_octave_id == 0)
		{
			if (_input_image_channels == 3)
			{
				if (_up_scale)
				{
					// ok = device.execute(taskResizeNN_color, offload, { (size_t)scaled_w,(size_t)scaled_h,1 }, { 1,1,1 }, 
					// img_dst_ui8, img_src_ui8, scaled_w, scaled_h, width, height, ratio_x, ratio_y);
					const int non_sampled_w = (float)i32w / _resample_ratio_x;
					const int non_sampled_h = (float)i32h / _resample_ratio_y;
					err = device.execute(_task_UpScale_3ch_8b, offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 },
						_imgColor_resampled, _imgColor_non_sampled, i32w, i32h, non_sampled_w, non_sampled_h, _resample_ratio_x, _resample_ratio_y);

					if (err != 0)return err;

					// __kernel void kColor_interleaved_ch(const global uchar* restrict inputImage,global float* restrict outputImage)
					err = device.execute(_task_Color, offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _imgColor_resampled, _scale_space_pyramid, i32w);
					if (err != 0)return err;
				}
				else
				{
					// __kernel void kColor_interleaved_ch(const global uchar* restrict inputImage,global float* restrict outputImage)
					err = device.execute(_task_Color, offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _imgColor_non_sampled, _scale_space_pyramid, i32w);
					if (err != 0)return err;
				}


			}
			else
			{
				if (_up_scale)
				{
					// ok = device.execute(taskResizeNN_color, offload, { (size_t)scaled_w,(size_t)scaled_h,1 }, { 1,1,1 }, 
					// img_dst_ui8, img_src_ui8, scaled_w, scaled_h, width, height, ratio_x, ratio_y);
					const int non_sampled_w = (float)i32w / _resample_ratio_x;
					const int non_sampled_h = (float)i32h / _resample_ratio_y;
					err = device.execute(_task_UpScale_1ch_32b, offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 },
						_scale_space_pyramid, _imgGray_non_sampled, i32w, i32h, non_sampled_w, non_sampled_h, _resample_ratio_x, _resample_ratio_y);

					if (err != 0)return err;
				}
			}
		}

		//Blur 
        for (int scl_id = 0; scl_id < _SIFT_MAX_SCALES; scl_id++)
		{
			const int i32fr = _filter_coef_widths[scl_id] / 2;

			if (scl_id < 1)
			{
				err = device.execute(*_task_BlurH[scl_id], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_space_pyramid, _imgFilters[scl_id], _scale_images_intermidiate[scl_id], i32w, i32h, i32fr, scl_id);
				if (err != 0)return err;

				err = device.execute(*_task_BlurV[scl_id], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_images_intermidiate[scl_id], _imgFilters[scl_id], _scale_space_pyramid, i32w, i32h, i32fr, scl_id);
				if (err != 0)return err;
			}
			else
			{
				//take a previous scale_space_image and use as an input!
				const auto ss_img_src_id = scl_id - 1;
				err = device.execute(*_task_BlurH[scl_id], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_space_pyramid, _imgFilters[scl_id], _scale_images_intermidiate[scl_id], i32w, i32h, i32fr, ss_img_src_id);
				if (err != 0)return err;

				err = device.execute(*_task_BlurV[scl_id], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_images_intermidiate[scl_id], _imgFilters[scl_id], _scale_space_pyramid, i32w, i32h, i32fr, scl_id);
				if (err != 0)return err;
			}
		}

		// Blur-differences	
        for (int i = 0; i < _SIFT_MAX_SCALES - 1; i++)
		{
			//__kernel void kDiff(const global float* restrict inputImage_next, const global float* restrict inputImage_prev, global float* restrict outputImage,const int w,const int ssid)			
			err = device.execute(*_task_Diff[i], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_space_pyramid, _diff_scale_images[i], i32w, i32h, i);
			if (err != 0)return err;
		}

		//Feature-detector
		const int i32ocatve_id = _octave_id;
		for (int i32scale_id = 1; i32scale_id <= _SIFT_INTVLS; i32scale_id++)
		{
			//__kernel void kDetector( global float4* restrict features,
			//const global float* restrict dog_p,const global float* restrict dog_c,const global float* restrict dog_n,
			//global uint* restrict count,const int w,const int h, const int s,const int o, const uint max_feat)		
			err = device.execute(*_task_Detector[i32scale_id - 1], offload,
			{ _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 },
				_kp_Detector,
				_diff_scale_images[i32scale_id - 1], _diff_scale_images[i32scale_id], _diff_scale_images[i32scale_id + 1],
				_counter_kp, i32w, i32h, i32scale_id, i32ocatve_id, _max_features);
			if (err != 0)return err;
		}

		return err;
	}
	
	int call_sync(const float offload,
        coopcl::virtual_device& device/*,
		std::unique_ptr<coopcl::clMemory>* scale_image_prev_octave = nullptr,
        coopcl::clTask* wait_task_prev_octave = nullptr*/)
	{
        int err = call_sync_detector(offload, device/*, scale_image_prev_octave, wait_task_prev_octave*/);
		if (err != 0)return err;

		const int i32w = _scale_image_width;
		const int i32h = _scale_image_height;

		const auto found_feats = found_features_cnt();
		if (found_feats < 1)return -1;
		const int i32found_feats = found_feats;

		//Feature-orientation
		const int i32octave_id = _octave_id;
		err = device.execute(*_task_Orientation, offload,
		{ found_feats*_SIFT_ORIENT_GPU_BLOCKSIZE,1,1 },
		{ _SIFT_ORIENT_GPU_BLOCKSIZE,1,1 },
			_kp_Detector, _scale_space_pyramid, i32w, i32h, i32found_feats, i32octave_id,_kp_Orientations);
		if (err != 0)return err;		
		
		//Feature descriptor
		err = device.execute(*_task_Descriptor, offload,
		{ found_feats*_SIFT_DESC_GPU_BLOCKSIZE,1,1 },
		{ _SIFT_DESC_GPU_BLOCKSIZE,1,1 },
			_kp_Detector, _kp_Orientations, _scale_space_pyramid, i32w, i32h, i32found_feats, i32octave_id, _kp_Descriptors);
		if (err != 0)return err;
		
		//Feature descriptor norm
		err = device.execute(*_task_Normalize, offload,
		{ found_feats,1,1 },{ 1,1,1 },_kp_Descriptors);
		if (err != 0)return err;
		
		return err;
	}

	int call_async_detector(const float offload,
		coopcl::virtual_device& device,
		std::unique_ptr<coopcl::clMemory>* scale_image_prev_octave = nullptr,
		coopcl::clTask* wait_task_prev_octave = nullptr)
	{

		int err = 0;

		err = reset(device, offload);
		if (err != 0)return err;

		const int i32w = _scale_image_width;
		const int i32h = _scale_image_height;

		const size_t gsx = i32w % 16 == 0 ? 16 : 1;
		const size_t gsy = i32h % 8 == 0 ? 8 : 1;

		if (_octave_id == 0)
		{
			if (_input_image_channels == 3)
			{
				if (_up_scale)
				{					
					// ok = device.execute(taskResizeNN_color, offload, { (size_t)scaled_w,(size_t)scaled_h,1 }, { 1,1,1 }, 
					// img_dst_ui8, img_src_ui8, scaled_w, scaled_h, width, height, ratio_x, ratio_y);
					const int non_sampled_w = (float)i32w / _resample_ratio_x;
					const int non_sampled_h = (float)i32h / _resample_ratio_y;
					err = device.execute_async(_task_UpScale_3ch_8b, offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 },
						_imgColor_resampled, _imgColor_non_sampled, i32w, i32h, non_sampled_w, non_sampled_h, _resample_ratio_x, _resample_ratio_y);

					if (err != 0)return err;

					// __kernel void kColor_interleaved_ch(const global uchar* restrict inputImage,global float* restrict outputImage)
					err = device.execute_async(_task_Color, offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _imgColor_resampled, _scale_space_pyramid, i32w);
					if (err != 0)return err;
				}
				else
				{
					// __kernel void kColor_interleaved_ch(const global uchar* restrict inputImage,global float* restrict outputImage)
					err = device.execute_async(_task_Color, offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _imgColor_non_sampled, _scale_space_pyramid, i32w);
					if (err != 0)return err;
				}

				
			}
			else
			{
				if (_up_scale)
				{
					// ok = device.execute(taskResizeNN_color, offload, { (size_t)scaled_w,(size_t)scaled_h,1 }, { 1,1,1 }, 
					// img_dst_ui8, img_src_ui8, scaled_w, scaled_h, width, height, ratio_x, ratio_y);
					const int non_sampled_w = (float)i32w / _resample_ratio_x;
					const int non_sampled_h = (float)i32h / _resample_ratio_y;
					err = device.execute_async(_task_UpScale_1ch_32b, offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 },
						_scale_space_pyramid, _imgGray_non_sampled, i32w, i32h, non_sampled_w, non_sampled_h, _resample_ratio_x, _resample_ratio_y);

					if (err != 0)return err;
				}
			}
		}
		else
		{
			if (scale_image_prev_octave == nullptr)return -111;
			if (wait_task_prev_octave == nullptr)return -222;


			_task_Down_pyramid.add_dependence(wait_task_prev_octave);

			const int wo = (int)(_scale_image_width * 2);
			const int d = (int)2;
			const int src_ss_image_id = _SIFT_INTVLS;
			// __kernel void kDown(global float* restrict dst, const global float* restrict src, const int wo, const int d,const int w, const int h)
			err = device.execute_async(_task_Down_pyramid, offload, { _scale_image_width, _scale_image_height,1 }, { gsx,gsy,1 }, _scale_space_pyramid, *scale_image_prev_octave, wo, d, i32w, i32h,src_ss_image_id);
			if (err != 0)return err;
		}

		//Blur 
		for (int scl_id = 0; scl_id < _SIFT_MAX_SCALES; scl_id++)
		{
			const int i32fr = _filter_coef_widths[scl_id] / 2;

			if (scl_id < 1)
			{
				err = device.execute_async(*_task_BlurH[scl_id], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_space_pyramid, _imgFilters[scl_id], _scale_images_intermidiate[scl_id], i32w, i32h, i32fr, scl_id);
				if (err != 0)return err;

				err = device.execute_async(*_task_BlurV[scl_id], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_images_intermidiate[scl_id], _imgFilters[scl_id], _scale_space_pyramid, i32w, i32h, i32fr, scl_id);
				if (err != 0)return err;
			}
			else
			{
				//take a previous scale_space_image and use as an input!
				const auto ss_img_src_id = scl_id - 1;
				err = device.execute_async(*_task_BlurH[scl_id], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_space_pyramid, _imgFilters[scl_id], _scale_images_intermidiate[scl_id], i32w, i32h, i32fr, ss_img_src_id);
				if (err != 0)return err;

				err = device.execute_async(*_task_BlurV[scl_id], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_images_intermidiate[scl_id], _imgFilters[scl_id], _scale_space_pyramid, i32w, i32h, i32fr, scl_id);
				if (err != 0)return err;
			}
		}

		// Blur-differences	
		for (int i = 0; i < _SIFT_MAX_SCALES - 1; i++)
		{
			//__kernel void kDiff(const global float* restrict inputImage_next, const global float* restrict inputImage_prev, global float* restrict outputImage,const int w,const int ssid)			
			err = device.execute_async(*_task_Diff[i], offload, { _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 }, _scale_space_pyramid, _diff_scale_images[i], i32w, i32h, i);
			if (err != 0)return err;
		}

		//Feature-detector
		const int i32ocatve_id = _octave_id;
		for (int i32scale_id = 1; i32scale_id <= _SIFT_INTVLS; i32scale_id++)
		{
			//__kernel void kDetector( global float4* restrict features,
			//const global float* restrict dog_p,const global float* restrict dog_c,const global float* restrict dog_n,
			//global uint* restrict count,const int w,const int h, const int s,const int o, const uint max_feat)		
			err = device.execute_async(*_task_Detector[i32scale_id - 1], offload,
			{ _scale_image_width,_scale_image_height,1 }, { gsx,gsy,1 },
				_kp_Detector,
				_diff_scale_images[i32scale_id - 1], _diff_scale_images[i32scale_id], _diff_scale_images[i32scale_id + 1],
				_counter_kp, i32w, i32h, i32scale_id, i32ocatve_id, _max_features);
			if (err != 0)return err;
		}

		return err;
	}

	int call_async(const float offload,
		coopcl::virtual_device& device,
		std::unique_ptr<coopcl::clMemory>* scale_image_prev_octave = nullptr,
		coopcl::clTask* wait_task_prev_octave = nullptr)
	{
		int err = call_async_detector(offload, device, scale_image_prev_octave, wait_task_prev_octave);
		if (err != 0)return err;

		const int i32w = _scale_image_width;
		const int i32h = _scale_image_height;

		const auto found_feats = found_features_cnt();
		if (found_feats < 1)return -1;
		const int i32found_feats = found_feats>_max_features?_max_features:found_feats;				
	
		//Feature-orientation
		const int i32octave_id = _octave_id;
		err = device.execute_async(*_task_Orientation, offload,
		{ found_feats*_SIFT_ORIENT_GPU_BLOCKSIZE,1,1 },
		{ _SIFT_ORIENT_GPU_BLOCKSIZE,1,1 },
			_kp_Detector, _scale_space_pyramid, i32w, i32h, i32found_feats,i32octave_id, _kp_Orientations);
		if (err != 0)return err;		

		//Feature descriptor
		err = device.execute_async(*_task_Descriptor, offload,
		{ found_feats*_SIFT_DESC_GPU_BLOCKSIZE,1,1 },
		{ _SIFT_DESC_GPU_BLOCKSIZE,1,1 },
			_kp_Detector, _kp_Orientations, _scale_space_pyramid, i32w, i32h, i32found_feats,i32octave_id, _kp_Descriptors);
		if (err != 0)return err;

		//Feature descriptor norm
		err = device.execute_async(*_task_Normalize, offload,
		{ found_feats,1,1 },{ 1,1,1 },_kp_Descriptors);
		if (err != 0)return err;

		return err;
	}

	public:

    sift_octave(const sift_octave&) = delete;
    sift_octave& operator=(const sift_octave&) = delete;
    sift_octave& operator=(sift_octave&& ) = delete;
    sift_octave(sift_octave&&) = delete;
    ~sift_octave()=default;
	sift_octave(
		const size_t id,
		const size_t w, const size_t h,
		const size_t color_depth,
		const std::map<std::string, std::string>& tasks,// map<func_name,func_body>
		coopcl::virtual_device& device,
		const bool async_call,
		const std::uint8_t* input_image = nullptr,
        const bool up_scale=false)
	{				
		_octave_id = id;		
		_input_image_channels = color_depth;
		_async_calls = async_call;
		_up_scale = up_scale;			

		if (_up_scale) 
		{
			_scale_image_height = _resample_ratio_y *h / std::pow(2, _octave_id);
			_scale_image_width = _resample_ratio_x *w / std::pow(2, _octave_id);
		}
		else 
		{
			_scale_image_height = h / std::pow(2, _octave_id);
			_scale_image_width = w / std::pow(2, _octave_id);
		}

		std::cout << "Build octave " << _octave_id + 1 <<", scale_image_size {"<<_scale_image_width<<","<<_scale_image_height<<"} pixels ..."<<std::endl;

		const size_t items = _scale_image_width*_scale_image_height;		
		
		int err = 0;
		err = Build_separable_filter_cof(_SIFT_SIGMA);
		if (err != 0)throw std::runtime_error("FilterCoeficients fail -> fixme!!");
		
		for(int i=0;i<_SIFT_MAX_SCALES;i++)
			_imgFilters.push_back(device.alloc<float>(_scale_filters[i],true));		
		
		_scale_space_pyramid = device.alloc<float>(items*_SIFT_MAX_SCALES);

		if (_octave_id == 0)
		{
			if (input_image == nullptr)throw std::runtime_error("Input_Image empty ? fixme!!");

			/* //Old code without up_sample
			if(color_depth>1)
				_imgColor = device.alloc<std::uint8_t>(color_depth * items, input_image);
			else			
				copy_pixels_1ch8_to_1ch32(input_image,_scale_image_width,_scale_image_height);			
				*/

			// Use (w,h) variables, not (_scale_image_width,_scale_image_height)
			// because imgColor or input_image is not yet resampled
			if (color_depth > 1)
			{
				_imgColor_non_sampled = device.alloc<std::uint8_t>(color_depth * w * h, input_image);
				_imgColor_resampled = device.alloc<std::uint8_t>(color_depth * _scale_image_width * _scale_image_height);
			}
			else
			{
				_imgGray_non_sampled = device.alloc<float>(w * h);
				copy_pixels_1ch8_to_1ch32(input_image, w, h);
			}
		}		
		
		//Detector_memory		
		_max_features = items / 100;
		_kp_Detector = device.alloc<cl_float4>(_max_features);		
		
		//Counter kp
		std::vector<std::uint32_t> dummy(1, 0);
		_counter_kp = device.alloc<std::uint32_t>(dummy); 

		//Orientation and descriptor memory 
		_kp_Orientations = device.alloc<float>(_max_features);
		_kp_Descriptors = device.alloc<float>(_max_features*_SIFT_DESC_GPU_BLOCKSIZE);
	
		
		std::stringstream jit;
		jit << "-cl-unsafe-math-optimizations "; //This option includes the -cl-no-signed-zeros and -cl-mad-enable options see OpenCL spec.
		jit << "-cl-fast-relaxed-math ";

		jit << "-D _SIFT_SIGMA=" << _SIFT_SIGMA;		
		jit << " -D _SIFT_IMG_BORDER=" << _SIFT_IMG_BORDER;
		jit << " -D _SIFT_CONTR_THR_1=" << _SIFT_CONTR_THR_1;
		jit << " -D _SIFT_MAX_INTERP_STEPS=" << _SIFT_MAX_INTERP_STEPS;
		jit << " -D _SIFT_INTVLS=" << _SIFT_INTVLS;
		jit << " -D _SIFT_CURV_THR=" << _SIFT_CURV_THR;
		jit << " -D _SIFT_CONTR_THR=" << _SIFT_CONTR_THR;		

		//jit << " -D _SIFT_ORI_EXTRA_CNT=" << _SIFT_ORI_EXTRA_CNT;
		jit << " -D _SIFT_ORI_HIST_BINS=" << _SIFT_ORI_HIST_BINS;
		jit << " -D _SIFT_ORIENT_GPU_BLOCKSIZE=" << _SIFT_ORIENT_GPU_BLOCKSIZE;
		jit << " -D _ORIENT_INIT_CONST=" << _ORIENT_INIT_CONST;
		jit << " -D _SIFT_ORI_SIG_FCTR=" << _SIFT_ORI_SIG_FCTR;  
		jit << " -D _SIFT_ORI_RADIUS=" << _SIFT_ORI_RADIUS;  
		jit << " -D _SIFT_ORI_PEAK_RATIO=" << _SIFT_ORI_PEAK_RATIO; 

		jit << " -D _SIFT_DESCR_HIST_CELLS=" << _SIFT_DESCR_HIST_CELLS; 
		jit << " -D _SIFT_DESCR_HIST_BINS=" << _SIFT_DESCR_HIST_BINS; 
		jit << " -D _SIFT_DESC_GPU_BLOCKSIZE=" << _SIFT_DESC_GPU_BLOCKSIZE;

		jit << " -D _SIFT_DESCR_SCL_FCTR=" << _SIFT_DESCR_SCL_FCTR;
		jit << " -D _SIFT_DESCR_MAG_THR=" << _SIFT_DESCR_MAG_THR;
		jit << " -D _SIFT_INT_DESCR_FCTR=" << _SIFT_INT_DESCR_FCTR;
		
		//const auto js = jit.str();	
		if (_octave_id == 0)
		{
			if (_input_image_channels == 3)
			{
                err = device.build_task(_task_Color,  tasks.at("kColor"), "kColor_interleaved_ch", jit.str());
				if (err != 0)throw std::runtime_error("kColor fail -> fixme!!");
			}

			if (_up_scale)
			{
				if (_input_image_channels == 3)
				{
                    err = device.build_task(_task_UpScale_3ch_8b,  tasks.at("kUp"), "kResize_NN_3ch_8b", jit.str());
					if (err != 0)throw std::runtime_error("kResize_NN_3ch_8b fail -> fixme!!");
					_task_Color.add_dependence(&_task_UpScale_3ch_8b);

				}
				else
				{
                    err = device.build_task(_task_UpScale_1ch_32b, tasks.at("kUp"), "kResize_NN_1ch_32b", jit.str());
					if (err != 0)throw std::runtime_error("kResize_NN_1ch_8b fail -> fixme!!");					
				}
			}			
		}
		else
		{			
            err = device.build_task(_task_Down_pyramid, tasks.at("kDown"), "kDown_pyramid", jit.str());
			if (err != 0)throw std::runtime_error("kDown fail -> fixme!!");
		}		

        for (size_t scale_id = 0; scale_id < _SIFT_MAX_SCALES; scale_id++)
		{
			_task_BlurH.push_back(std::make_unique<coopcl::clTask>());
            err = device.build_task(*_task_BlurH[scale_id], tasks.at("kBlur"), "kBlurH", jit.str());
			if (err != 0)if (err != 0)if (err != 0)throw std::runtime_error("kBlur fail -> fixme!!");

			_task_BlurV.push_back(std::make_unique<coopcl::clTask>());
            err = device.build_task(*_task_BlurV[scale_id], tasks.at("kBlur"), "kBlurV", jit.str());
			if (err != 0)if (err != 0)if (err != 0)throw std::runtime_error("kBlur fail -> fixme!!");

			if (async_call)
			{
				// now set task_dependencies			
				if (scale_id > 0)
				{
					_task_BlurH[scale_id]->add_dependence(_task_BlurV[scale_id - 1].get());
					_task_BlurV[scale_id]->add_dependence(_task_BlurH[scale_id].get());
				}
				else
				{
					if (_octave_id == 0)
					{
						if (_input_image_channels == 3)
						{
							_task_BlurH[scale_id]->add_dependence(&_task_Color);
						}
						else
						{
							if (_up_scale) _task_BlurH[scale_id]->add_dependence(&_task_UpScale_1ch_32b);
						}
					}
					else
					{						
						_task_BlurH[scale_id]->add_dependence(&_task_Down_pyramid);
					}

					_task_BlurV[scale_id]->add_dependence(_task_BlurH[scale_id].get());
				}
			}

			//now memory tmp
			_scale_images_intermidiate.push_back(device.alloc<float>(items));
		}

		for (int i = 0; i < _SIFT_MAX_SCALES -1; i++)
		{
			_task_Diff.push_back(std::make_unique<coopcl::clTask>());			
            err = device.build_task(*_task_Diff[i], tasks.at("kDiff"), "kDiff", jit.str());
			if (err != 0)throw std::runtime_error("kDiff fail -> fixme!!");
			
			//set task_dependencies
			if (async_call)
				_task_Diff[i]->add_dependence(_task_BlurV[i + 1].get());

			//now memory
			_diff_scale_images.push_back(device.alloc<float>(items));
		}		
		
		for (int i = 0; i < _SIFT_INTVLS; i++)
		{
			_task_Detector.push_back(std::make_unique<coopcl::clTask>());
            err = device.build_task(*_task_Detector[i], tasks.at("kDetector"), "kDetector", jit.str());
			if (err != 0)throw std::runtime_error("kDetector fail -> fixme!!");			
		}

		//set dependencies
		if (async_call)
		{
			for (size_t i = 1; i <= _SIFT_INTVLS; i++)
			{
				_task_Detector[i - 1]->add_dependence(_task_Diff[i - 1].get());
				_task_Detector[i - 1]->add_dependence(_task_Diff[i].get());
				_task_Detector[i - 1]->add_dependence(_task_Diff[i + 1].get());
			}
		}

        err = device.build_task(_task_Reset, tasks.at("kReset"), "kReset", jit.str());
		if (err != 0)throw std::runtime_error("kReset fail -> fixme!!");

		if (async_call)
		{
            for (size_t i = 0; i < _SIFT_INTVLS; i++)
				_task_Reset.add_dependence(_task_Detector[i].get());
		}
		
		//-------------------------------------------------------------------
		//Build orientation,descriptor, norm and set dependencies
		//-------------------------------------------------------------------
		_task_Orientation = std::make_unique<coopcl::clTask>();
        err = device.build_task(*_task_Orientation, tasks.at("kOrientation"), "kOrientation", jit.str());
		if (err != 0)throw std::runtime_error("kOrientation fail -> fixme!!");

		if (async_call)
		{
			for (int i = 0; i < _SIFT_INTVLS; i++)
				_task_Orientation->add_dependence(_task_Detector[i].get());
		}

		_task_Descriptor = std::make_unique<coopcl::clTask>();
        err = device.build_task(*_task_Descriptor, tasks.at("kDescriptor"), "kDescriptor", jit.str());
		if (err != 0)throw std::runtime_error("kDescriptor fail -> fixme!!");
		
		if (async_call)		
			_task_Descriptor->add_dependence(_task_Orientation.get());

		_task_Normalize = std::make_unique<coopcl::clTask>();
        err = device.build_task(*_task_Normalize, tasks.at("kNormalize"), "kNormalize", jit.str());
		if (err != 0)throw std::runtime_error("kNormalize fail -> fixme!!");
		
		if (async_call)
			_task_Normalize->add_dependence(_task_Descriptor.get());
	}

	int update_pixels(const std::uint8_t* input_image, const size_t w, const size_t h,const size_t color_depth)
	{
		if (input_image == nullptr)throw std::runtime_error("In_Image empty ? fixme!!");
		if (color_depth == 3)
		{
			if (_imgColor_non_sampled == nullptr)throw std::runtime_error("Color_Image empty ? fixme!!");
			const auto items = w*h;
			std::memcpy(_imgColor_non_sampled->data(), input_image, color_depth * items);
		}
		else
			copy_pixels_1ch8_to_1ch32(input_image, w, h);
		return 0;
	}

	int call_detector_descriptor(const float offload,
		coopcl::virtual_device& device,
		std::unique_ptr<coopcl::clMemory>* scale_image_prev_octave = nullptr,
		coopcl::clTask* wait_task_prev_octave = nullptr)
	{
		if (_async_calls)
			return call_async(offload, device, scale_image_prev_octave, wait_task_prev_octave);
		
        return call_sync(offload, device/*, scale_image_prev_octave, wait_task_prev_octave*/);
	}

	int call_detector(const float offload,
		coopcl::virtual_device& device,
		std::unique_ptr<coopcl::clMemory>* scale_image_prev_octave = nullptr,
		coopcl::clTask* wait_task_prev_octave = nullptr)
	{
		if (_async_calls)
			return call_async_detector(offload, device, scale_image_prev_octave, wait_task_prev_octave);

        return call_sync_detector(offload, device/*, scale_image_prev_octave, wait_task_prev_octave*/);
	}

	int wait()const 
	{	
		int err = 0;
		
		for (int i = 0; i < _SIFT_INTVLS; i++)
		{
			err = _task_Detector[i]->wait();
			if (err != 0)return err;
		}

		return err;
	}

#ifdef _DEBUG
	const float* get_gray()const
	{
		const auto err = _task_Color.wait();
		if (err != 0)return nullptr;
		return (float*)_scale_space_pyramid->data();
	}

	const float* get_blur(const size_t id)const {
		std::string img_name;
		std::cout << "visualize ..." << std::endl;
		const float* pimg = nullptr;

		if (id >= _SIFT_MAX_SCALES)return pimg;
		auto pf = ((float*)_scale_space_pyramid->data());
		pimg = pf + id*_scale_image_width*_scale_image_height;

		img_name = "Blur";
		img_name.append(std::to_string(id + 1));
		return pimg;
	}

	const float* get_diff_blur(const size_t id)const {
		std::string img_name;
		std::cout << "visualize ..." << std::endl;
		float* pimg = nullptr;

		if (id >= _SIFT_MAX_SCALES - 1)return pimg;
		pimg = (float*)_diff_scale_images[id]->data();

		img_name = "BlurDiff";
		img_name.append(std::to_string(id + 1));
		return pimg;
	}
#endif // _DEBUG

	std::vector<float> descriptors()const
	{
		auto err = _task_Normalize->wait();
		if (err != 0)return{};
		
		const auto found_feats = found_features_cnt();

		std::vector<float> descriptors(found_feats*_SIFT_DESC_GPU_BLOCKSIZE, 0);
		for (size_t i = 0; i < descriptors.size(); i++)
			descriptors[i] = _kp_Descriptors->at<cl_float>(i);	
		
		return descriptors;
	}

	std::vector<float> orientations()const
	{
		auto err = _task_Orientation->wait();
		if (err != 0)return{};

		const auto found_feats = found_features_cnt();
		
		std::vector<float> orientations(found_feats, 0);
		for (size_t i = 0; i < orientations.size(); i++)
			orientations[i] = _kp_Orientations->at<cl_float>(i);

		return orientations;
	}

	std::vector<cl_float4> features()const
	{
		int err = 0;
		std::vector<cl_float4> feats;		
		const auto found_feats = found_features_cnt();
		
		for (size_t i = 0; i < found_feats; i++)
		{
			const auto feature = _kp_Detector->at<cl_float4>(i);
			if (feature.x > -1 && feature.y > -1 && feature.w >-1) 			
			feats.push_back(feature);			
		}
		return feats;
	}	
	
	std::unique_ptr<coopcl::clMemory>* imgBlur() { return &_scale_space_pyramid; }

	coopcl::clTask* task_Blur() { 
		return _task_BlurV[_SIFT_INTVLS].get();
	}

	void store_task_records_in_files(const std::string& path, const size_t oid)const
	{				
		//sid=scale_id, oid=octave_id
		size_t sid = 0;
		store_task_in_file(&_task_Reset, path, oid, sid);
		store_task_in_file(&_task_Color, path, oid, sid);
		store_task_in_file(&_task_Down_pyramid, path, oid, sid);

		for (const auto& t : _task_BlurH)
			store_task_in_file(t.get(), path, oid, sid++);

		sid = 0;
		for (const auto& t : _task_BlurV)
			store_task_in_file(t.get(), path, oid, sid++);

		sid = 0;
		for (const auto& t : _task_Diff)
			store_task_in_file(t.get(), path, oid, sid++);

		sid = 0;
		for (const auto& t : _task_Detector)
			store_task_in_file(t.get(), path, oid, sid++);
		
		sid = 0;
		store_task_in_file(_task_Orientation.get(), path, oid, sid);
		store_task_in_file(_task_Descriptor.get(), path, oid, sid);
		store_task_in_file(_task_Normalize.get(), path, oid, sid);		

		return;
	}
};

class sift_algorithm
{

private:
	
	const static auto _CNT_OCTAVES = _SIFT_MAX_OCTAVES;	

	coopcl::virtual_device device;
	std::map<std::string,std::string> sift_tasks;
	std::vector<std::unique_ptr<sift_octave>> octaves;

	int img_width{ 0 };
	int img_height{ 0 };

public:
	
	static size_t cnt_octaves(){ return _SIFT_MAX_OCTAVES;}
	static size_t cnt_scales(){ return _SIFT_MAX_SCALES; }

    sift_algorithm(const sift_algorithm&) = delete;
    sift_algorithm& operator=(const sift_algorithm&) = delete;
    sift_algorithm& operator=(sift_algorithm&& ) = delete;
    sift_algorithm(sift_algorithm&&) = delete;
    ~sift_algorithm()= default;

	sift_algorithm(
		const unsigned char* input_image,
		const int width,
		const int height,
		const int color_depth,
		const bool up_scale=false)
	{
		sift_tasks.emplace("kColor",SIFT_tasks::task_Color);		//kColor
		sift_tasks.emplace("kDown",SIFT_tasks::task_Down);			//kDown (resample)
		sift_tasks.emplace("kUp", SIFT_tasks::task_Resize);			//kUp (resample)
		sift_tasks.emplace("kBlur",SIFT_tasks::task_Blur);			//kBlur
		sift_tasks.emplace("kDiff", SIFT_tasks::task_Diff);				//kDiff
		sift_tasks.emplace("kDetector", SIFT_tasks::task_Detector);		//kDetector
		sift_tasks.emplace("kReset", SIFT_tasks::task_Reset);				//kReset
		
		// tasks Orientation, Descriptor and Normalize needs some defines
		// create this tasks, combine together tasks and defines
		std::string defs_orient = SIFT_tasks::task_defines;
		defs_orient.append(SIFT_tasks::task_Orientation);
		sift_tasks.emplace("kOrientation", defs_orient);	//kOrientation

		std::string defs_descriptor = SIFT_tasks::task_defines;
		defs_descriptor.append(SIFT_tasks::task_Descriptor);
		sift_tasks.emplace("kDescriptor", defs_descriptor);		//kDescriptor

		std::string defs_norm = SIFT_tasks::task_defines;
		defs_norm.append(SIFT_tasks::task_Normalize);
		sift_tasks.emplace("kNormalize", defs_norm);		//kNormalize		
		
		for (int i = 0; i < _CNT_OCTAVES; i++)
		{
			if (i == 0)
                octaves.push_back(std::make_unique<sift_octave>(i, width, height, color_depth, sift_tasks, device, true, input_image,up_scale));
			else
                octaves.push_back(std::make_unique<sift_octave>(i, width, height, color_depth, sift_tasks, device, true, nullptr, up_scale));
		}

		img_height = height;
		img_width = width;
	}

#ifdef _DEBUG
	const float* get_gray(const float offload, const size_t oid, const size_t color_channels = 1)
	{
		int err = 0;

		if (color_channels == 3 || oid > 0)
		{
			for (size_t oct_id = 0; oct_id < octaves.size(); oct_id++)
			{
				if (oct_id == 0)
				{
					err = octaves[oct_id]->call_detector(offload, device);
					if (err != 0) return nullptr;
				}
				else
				{
					err = octaves[oct_id]->call_detector(offload, device, octaves[oct_id - 1]->imgBlur(), octaves[oct_id - 1]->task_Blur());
					if (err != 0) return nullptr;
				}
			}

			for (auto const& octave : octaves)
			{
				err = octave->wait();
				if (err != 0) return nullptr;
			}
		}

		return octaves.at(oid)->get_gray();
	}

	const float* get_blur(const float offload, const size_t sid, const size_t oid)
	{
		int err = 0;
		for (size_t oct_id = 0; oct_id < octaves.size(); oct_id++)
		{
			if (oct_id == 0)
			{
				err = octaves[oct_id]->call_detector(offload, device);
				if (err != 0) return nullptr;
			}
			else
			{
				err = octaves[oct_id]->call_detector(offload, device, octaves[oct_id - 1]->imgBlur(), octaves[oct_id - 1]->task_Blur());
				if (err != 0) return nullptr;
			}
		}

		for (auto const& octave : octaves)
		{
			err = octave->wait();
			if (err != 0) return nullptr;
		}
		return octaves.at(oid)->get_blur(sid);
	}

	const float* get_diff_blur(const float offload, const size_t sid, const size_t oid)
	{
		int err = 0;
		for (size_t oct_id = 0; oct_id < octaves.size(); oct_id++)
		{
			if (oct_id == 0)
			{
				err = octaves[oct_id]->call_detector(offload, device);
				if (err != 0) return nullptr;
			}
			else
			{
				err = octaves[oct_id]->call_detector(offload, device, octaves[oct_id - 1]->imgBlur(), octaves[oct_id - 1]->task_Blur());
				if (err != 0) return nullptr;
			}
		}

		for (auto const& octave : octaves)
		{
			err = octave->wait();
			if (err != 0) return nullptr;
		}

		return octaves.at(oid)->get_diff_blur(sid);
	}

#endif // _DEBUG

	int extract_features(
		size_t& elapsed_time,
		const float offload,
		std::vector <SIFT::sfeature> & features,
		std::vector<float>& descriptors, 
		std::vector<float>& orientations)
	{
		int err = 0;
		const auto start = std::chrono::system_clock::now();

		for (size_t oct_id = 0; oct_id < octaves.size(); oct_id++)
		{
			if (oct_id == 0)
			{
				err = octaves[oct_id]->call_detector_descriptor(offload, device);
				if (err != 0) return err;
			}
			else
			{
				err = octaves[oct_id]->call_detector_descriptor(offload, device, octaves[oct_id - 1]->imgBlur(), octaves[oct_id - 1]->task_Blur());
				if (err != 0) return err;
			}
		}

		for (auto const& octave : octaves)
		{
			err = octave->wait();
			if (err != 0) return err;
		}

		const auto end = std::chrono::system_clock::now();
		const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		elapsed_time = et;
		size_t oid = 0;
		for (auto const& octave : octaves)
		{
			const auto feats = octave->features();
			for (const auto f : feats)
			{
				SIFT::sfeature ff{};
				ff.x = f.x;
				ff.y = f.y;
				ff.sigma = f.z;
				ff.scale_id = f.w;
				ff.octave_id = oid;
				features.push_back(ff);
			}
			oid++;
			const auto desc = octave->descriptors();
			for (const auto d : desc)
				descriptors.push_back(d);

			const auto orients = octave->orientations();
			for (const auto o : orients)
				orientations.push_back(o);
		}
		std::cout << "--------------------------------------------------------" << std::endl;
		std::cout << "Elapsed_time_ms," << et << std::endl;
		std::cout << "--------------------------------------------------------" << std::endl;
		return err;
	}

	int extract_features(
		size_t& elapsed_time,
		const float offload,
		std::vector<SIFT::sfeature>& features)
	{
		int err = 0;
		const auto start = std::chrono::system_clock::now();

		for (size_t oct_id = 0; oct_id < octaves.size(); oct_id++)
		{
			if (oct_id == 0)
			{
				err = octaves[oct_id]->call_detector(offload, device);
				if (err != 0) return err;
			}
			else
			{
				err = octaves[oct_id]->call_detector(offload, device, octaves[oct_id - 1]->imgBlur(), octaves[oct_id - 1]->task_Blur());
				if (err != 0) return err;
			}
		}

		for (auto const& octave : octaves)
		{
			err = octave->wait();
			if (err != 0) return err;
		}

		const auto end = std::chrono::system_clock::now();
		const auto et = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();		
		elapsed_time = et;
		size_t oid = 0;
		for (auto const& octave : octaves)
		{			
			const auto feats = octave->features();
			for (const auto f : feats)
			{
				SIFT::sfeature ff{};
				ff.x = f.x;
				ff.y = f.y;
				ff.sigma = f.z;
				ff.scale_id = f.w;
				ff.octave_id = oid;
				features.push_back(ff);
			}
			oid++;
		}
		std::cout << "--------------------------------------------------------" << std::endl;		
		std::cout << "Elapsed_time_ms," << et << std::endl;
		std::cout << "--------------------------------------------------------" << std::endl;
		return err;
	}
	
	int update_pixels(
		const unsigned char* input_image,
		const int width,
		const int height,
		const int color_depth) 
	{
		return octaves.begin()->get()->update_pixels(input_image, width, height, color_depth);
	}

	void store_task_exec_times(const std::string& path)const 
	{
		size_t oid = 0;
		for (const auto& o : octaves) 		
			o->store_task_records_in_files(path, oid++);		
	}

    int image_width()const { return img_width; }
    int image_height()const { return img_height; }

};

static std::unique_ptr<sift_algorithm> sift{nullptr};

namespace SIFT
{
#ifdef _DEBUG

	size_t cnt_scales() { return sift_algorithm::cnt_scales(); }

	size_t cnt_octaves() { return sift_algorithm::cnt_octaves(); }

	const float* get_gray(
		const float offload,
		const unsigned char* input_color_image,
		const int width,
		const int height,
		const int color_depth,
		const int oid)
	{
		const auto items = width*height;

		if (input_color_image == nullptr) {
			std::cerr << "Load image failed, fixme!" << std::endl;
			exit(-1);
		}

		//check if algorithm_type is created?
		if (sift != nullptr)
		{
			//check if a frame with a different size was called ?
			if (width != sift->image_width() || height != sift->image_height())
				sift = std::make_unique<sift_algorithm>(input_color_image, width, height, color_depth);
			else //write new pixels
				sift->update_pixels(input_color_image, width, height, color_depth);			
		}
		else
			sift = std::make_unique<sift_algorithm>(input_color_image, width, height, color_depth);

		return sift->get_gray(offload, oid, color_depth);
	}

	const float* get_blur(const float offload,
		const unsigned char* input_color_image,
		const int width,
		const int height,
		const int color_depth,
		const int oid,
		const int sid)
	{
		const auto items = width*height;

		if (input_color_image == nullptr) {
			std::cerr << "Load image failed, fixme!" << std::endl;
			exit(-1);
		}

		//check if algorithm_type is created?
		if (sift != nullptr)
		{
			//check if a frame with a different size was called ?
			if (width != sift->image_width() || height != sift->image_height())
				sift = std::make_unique<sift_algorithm>(input_color_image, width, height, color_depth);
			else
			{
				//write pixels
				sift->update_pixels(input_color_image, width, height, color_depth);
			}
		}
		else
			sift = std::make_unique<sift_algorithm>(input_color_image, width, height, color_depth);

		return sift->get_blur(offload, sid, oid);
	}

	const float* get_diff_blur(const float offload, 
		const unsigned char* input_color_image,
		const int width, 
		const int height, 
		const int color_depth,
		const int oid, const int sid)
	{
		const auto items = width*height;

		if (input_color_image == nullptr) {
			std::cerr << "Load image failed, fixme!" << std::endl;
			exit(-1);
		}

		//check if algorithm_type is created?
		if (sift != nullptr)
		{
			//check if a frame with a different size was called ?
			if (width != sift->image_width() || height != sift->image_height())
				sift = std::make_unique<sift_algorithm>(input_color_image, width, height, color_depth);
			else
			{
				//write pixels
				sift->update_pixels(input_color_image, width, height, color_depth);
			}
		}
		else
			sift = std::make_unique<sift_algorithm>(input_color_image, width, height, color_depth);

		return sift->get_diff_blur(offload, sid, oid);
	}

	std::vector<std::uint8_t> test_kResize(const float offload,
		const unsigned char* input_color_image,
		const int width, const int height,
		const int color_depth,
		const float ratio_x,const float ratio_y)
	{
		coopcl::virtual_device device;
		coopcl::clTask taskResizeNN_color;
		
		auto ok = device.build_task(taskResizeNN_color, {}, SIFT_tasks::task_Resize, "kResize_NN_3ch_8b", "");		
		auto img_src_ui8 = device.alloc<std::uint8_t>(width * height*color_depth, input_color_image,true);						
		const int scaled_w = width*ratio_x;
		const int scaled_h = height*ratio_y;				

		auto img_dst_ui8 = device.alloc<std::uint8_t>(scaled_w * scaled_h*color_depth);
		ok = device.execute(taskResizeNN_color, offload, { (size_t)scaled_w,(size_t)scaled_h,1 }, { 1,1,1 }, img_dst_ui8, img_src_ui8,scaled_w,scaled_h, width,height,ratio_x,ratio_y);
		
		std::vector<std::uint8_t> img(scaled_w * scaled_h*color_depth);
		std::memcpy(img.data(), img_dst_ui8->data(), sizeof(std::uint8_t) * img.size());
		return img;
	}

	std::vector<float> test_ColorToGray_Resize(const float offload,
		const unsigned char* input_color_image,
		const int width, const int height,
		const int color_depth,
		const float ratio_x, const float ratio_y)
	{
		coopcl::virtual_device device;
		coopcl::clTask taskColor, taskResizeNN;

		auto ok = device.build_task(taskColor, {}, SIFT_tasks::task_Color, "kColor_interleaved_ch", "");
		ok = device.build_task(taskResizeNN, {}, SIFT_tasks::task_Resize, "kResize_NN_1ch_32b", "");
		
		auto img_src_ui8 = device.alloc<std::uint8_t>(width * height * color_depth, input_color_image, true);		
		auto img_src_f32 = device.alloc<float>(width * height, 0.0f);

		const int scaled_w = width * ratio_x;
		const int scaled_h = height * ratio_y;
		auto scaled_img_f32 = device.alloc<float>(scaled_w * scaled_h);

		ok = device.execute(taskColor, offload, { (size_t)width,(size_t)height,1 }, { 1,1,1 }, img_src_ui8, img_src_f32, width);
		ok = device.execute(taskResizeNN, offload, { (size_t)scaled_w,(size_t)scaled_h,1 }, { 1,1,1 }, scaled_img_f32, img_src_f32, scaled_w, scaled_h, width, height, ratio_x, ratio_y);

		std::vector<float> img(scaled_img_f32->items());
		std::memcpy(img.data(), scaled_img_f32->data(), sizeof(float) * img.size());
		return img;
	}

#endif // _DEBUG

	void init(const unsigned char* input_image,
		const int width, const int height,
		const int color_depth,
		const bool up_scale)
	{
		const auto app_init_start = std::chrono::system_clock::now();

		if (input_image == nullptr) {
			std::cerr << "Load image failed, fixme!" << std::endl;
			exit(-1);
		}

		//check if algorithm_type is created?
		if (sift != nullptr)
		{
			//check if a frame with a different size was called ?
			if (width != sift->image_width() || height != sift->image_height())
			{
				std::cout << "Initialize ..." << std::endl;
				sift = std::make_unique<sift_algorithm>(input_image, width, height, color_depth,up_scale);
			}
			else
			{
				//write pixels,size is the same as for previous frame
				sift->update_pixels(input_image, width, height,color_depth);
			}
		}
		else
		{
			std::cout << "Initialize ..." << std::endl;
			sift = std::make_unique<sift_algorithm>(input_image, width, height, color_depth,up_scale);			
		}

		const auto app_init_end = std::chrono::system_clock::now();
		const auto et_init = std::chrono::duration_cast<std::chrono::milliseconds>(app_init_end - app_init_start).count();	

		//Call task_graph
		std::cout << "Execute ..." << std::endl;

		//pause for GPU-driver to increase pwr_measure accuracy!
		//std::this_thread::sleep_for(std::chrono::seconds(2));
	}

	int extract_features(
		size_t& elapsed_time,
		const float offload,
		const unsigned char* input_color_image,
		const int width, const int height, 
		const int color_depth,
		std::vector<sfeature>& features,
		std::vector<float>& descriptors,
		std::vector<float>& orientations,
		const bool up_scale)
	{
		init(input_color_image,width,height, color_depth, up_scale);
		return sift->extract_features(elapsed_time,offload, features, descriptors,orientations);
	}

	int extract_features(
		size_t& elapsed_time,
		const float offload,
		const unsigned char* input_color_image,
		const int width, const int height,
		const int color_depth,
		std::vector < SIFT::sfeature > & features,
		const bool up_scale)
	{
		init(input_color_image, width, height,color_depth,up_scale);
		return sift->extract_features(elapsed_time, offload, features);
	}

	void store_task_exec_times(const std::string& path)
	{        
		sift->store_task_exec_times(path);

		/*
		const auto fpath = "C:/Users/morkon/Sift/kDetector00_obs.csv";
		coopcl::clTask t;
		t.read_task_profile(fpath);*/
	}
}
