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

typedef struct{
    int point1;
    float x1;
    float y1;
    int point2;
    float x2;
    float y2;
    float dist;
    float orientationDiff;
    float scaleDiff;
    int match;
    bool dup;
} distPoint;

typedef struct{
    float x;
    float y;
    float scale;
    float orientation;
    int laplacian;
    int clusterIndex;
    float descriptor[64];
} Ipoint;

__kernel void NearestNeighbor(__global Ipoint *d_ipts1,
                              __global Ipoint *d_ipts2,
                              __global distPoint *d_distancePoints,
                              __local float* tempDistPts,
                              const unsigned int ipts2Size) {

    //int groupId = get_group_id(0);
	int groupId = ccl_get_group_id(0);
    int localId = get_local_id(0);
    int localSize = get_local_size(0);

    __global Ipoint *ipts1 = d_ipts1 + groupId;
    __global Ipoint *ipts2;

    __global distPoint *dp = d_distancePoints + groupId;

    int point2Match = 0;
    float minDist = FLT_MAX;

    for (unsigned int i2 = 0; i2 < ipts2Size; i2++){
         
        // Calculate the sum of squares distance
        ipts2 = d_ipts2 + i2;
        // TODO This metric may need to be improved.  Currently
        //      we determine matches based on the sum of descriptor
        //      differences
        tempDistPts[localId] = fabs(ipts1->descriptor[localId] -
                                    ipts2->descriptor[localId]);
        barrier(CLK_LOCAL_MEM_FENCE);
         
        // reduce
        for(int k=32; k > 0 ;k>>=1) {
            if(localId < k) 
            {
                tempDistPts[localId] += tempDistPts[localId + k];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if (localId==0) {
            tempDistPts[0] /= 64; // Get the average descriptor difference
            if(tempDistPts[0] < minDist) 
            {
                minDist = tempDistPts[0];
                point2Match = i2;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(localId == 0) 
    {
        ipts2 = d_ipts2+point2Match;
        dp->point1 = groupId;
        dp->x1 = ipts1->x;
        dp->y1 = ipts1->y;
        dp->point2 = point2Match;
        dp->x2 = ipts2->x;
        dp->y2 = ipts2->y;
        dp->dist = minDist;
        dp->orientationDiff = ipts1->orientation - ipts2->orientation;
        dp->scaleDiff = ipts1->scale - ipts2->scale;      
    }
}