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

#ifndef CVUTILS_H
#define CVUTILS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "surf.h"

typedef struct{
    // Point from the new frame
    int point1;
    float x1;
    float y1;
    // Point from the reference frame
    int point2;
    float x2;
    float y2;
    // Information about their relationship
    float dist;
    float orientationDiff;
    float scaleDiff;
    // TODO: If dup or match, drop the point? (These shouldn't have to
    //       be part of this data structure either way)
    int match;
    bool dup; 
} distPoint;

// Compute the distances between points in the reference image (used
// to determine the scaling factor
float** computeDistanceTable(IpVec* distancePoints);

// Create the windows required for the stabilization algorithm
void createStabilizationWindows(); 

//! Draw the FPS figure on the image (requires at least 2 calls)
void drawFPS( cv::Mat& img);

// Draw the new and reference images
int drawImages( cv::Mat& img1,  cv::Mat& img2,
               std::vector<distPoint>* distancePoints, float shakeThreshold, 
               float** distTable);

//! Draw a single feature on the image
void drawIpoint( cv::Mat& img, Ipoint &ipt);

//! Draw all the Ipoints in the provided vector
void drawIpoints( cv::Mat& img, std::vector<Ipoint> &ipts);

//! Draw the ipts as dots
void drawIpointsAsDots( cv::Mat& img, std::vector<Ipoint> &ipts);

//! Draw a Point at feature location
void drawPoint( cv::Mat& img, Ipoint &ipt);

//! Draw a Point at all features
void drawPoints( cv::Mat& img, std::vector<Ipoint> &ipts);

//! Draw descriptor windows around Ipoints in the provided vector
void drawWindows( cv::Mat& img, std::vector<Ipoint> &ipts);

//! Euclidian distance between points within a distPoint structure
float euclideanDist(distPoint p);

//! Euclidian distance between two pairs of coordinates
float euclideanDist(float x1, float y1, float x2, float y2);

//! Release the memory allocated for the distance table
void freeDistanceTable(float** distTable, unsigned int numPts);

//! Round float to nearest integer
inline int fRound(float flt)
{
    return (int) floor(flt+0.5f);
}

//! Convert image to single channel 32F
//IplImage* getGray(const  cv::Mat& img);
cv::Mat getGray(const  cv::Mat& img);

// Determine the rotation of an image with respect to the reference
float getRotation(std::vector<distPoint> distancePoints);

// Determine the scaling factor of an image with respect to the reference
float getScalingFactor(std::vector<distPoint> distancePoints, 
                       float** distTable, float rotation,
                       cv::Point2d imgCenter);

// Determine the translation (shift) of an image after rotating and
// scaling to match the reference
cv::Point2d getTranslation(std::vector<distPoint> distancePoints,
                            float scale, float rotation, 
                            cv::Point2d imgCenter);

//! Load the SURF features from file
void loadSurf(char *filename, std::vector<Ipoint> &ipts);

// Sort an array and return the median value
float median(std::vector<float> v);

IpVec* mergeIpts(IpVec latestPts, IpVec prevPts, std::vector<distPoint> &distPoints); 

//! Save the SURF-ified image with a -surf.jpg extension
void saveImage(const  cv::Mat& img, char *originalName);

//! Show the provided image and wait for keypress
void showImage(const  cv::Mat& img, bool blocking=true);

//! Show the provided image in titled window and wait for keypress
void showImage(char *title, const  cv::Mat& img, bool blocking=true);

//! Write Text on Image
void writeImageName( cv::Mat& img, char * s);

//! Write on Top of Image
void writeImageText( cv::Mat& img, char * s);

//! Save the SURF features to file
void writeIptsToFile(char *path, std::vector<Ipoint> &ipts);

//! Save the distance data to file
void writeDptsToFile(char *path, std::vector<distPoint> &dpts);

#endif
