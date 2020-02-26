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

#include <iostream>
#include <fstream>
#include <time.h>
#include <limits>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "cvutils.h"
#include "utils.h"

static const int NCOLOURS = 8;
static const cv::Scalar COLOURS[] = {cv::Scalar(255,0,0), cv::Scalar(0,255,0),
                                   cv::Scalar(0,0,255), cv::Scalar(255,255,0),
                                   cv::Scalar(0,255,255), cv::Scalar(255,0,255),
                                   cv::Scalar(255,255,255), cv::Scalar(0,0,0)};

//! The identify how scaled the image is, we need to compare points
//  within the reference image and within the new image and see the 
//  difference.  To do that, we should create a reference table for
//  the reference points (this only has to be done once)
//  TODO: Make this more efficient
float** computeDistanceTable(IpVec* distancePoints) 
{
    int numRefPoints = distancePoints->size();

    float** refTable = NULL;
    
    refTable = (float**)alloc(sizeof(float*)*numRefPoints);

    for(int i = 0; i < numRefPoints; i++) {
        refTable[i] = (float*)alloc(sizeof(float)*numRefPoints);
    }

    // Not efficient
    for(int i = 0; i < numRefPoints; i++) {
        float myX = distancePoints->at(i).x;
        float myY = distancePoints->at(i).y;
        for(int j = 0; j < numRefPoints; j++) {
            float otherX = distancePoints->at(j).x;
            float otherY = distancePoints->at(j).y;
            refTable[i][j] = euclideanDist(myX, myY, otherX, otherY);
        }
    }

    return refTable;
}

// This only needs to be done once (otherwise it will leak memory)
void createStabilizationWindows() {
    
    cv::namedWindow("Orig Frame", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("New Frame", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Output Frame", cv::WINDOW_AUTOSIZE);

    // Uncomment the following lines to see intermediate images
    /*
    // Rotated image
    cvNamedWindow("Rotated Frame", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Scaled Frame", CV_WINDOW_AUTOSIZE);
    */
}

//! Draw the FPS figure on the image (requires at least 2 calls)
void drawFPS( cv::Mat& img)
{
    static int counter = 0;
    static clock_t t;
    static float fps = 0;
    char fps_text[20];
    
    //cv::Font font;
    //cv::InitFont(&font, cv::FONT_HERSHEY_SIMPLEX| cv::FONT_ITALIC, 0.5, 0.5, 0, 1);

    // Add fps figure (every 10 frames)
    if (counter > 10)
    {
        fps = (10.0f/(clock()-t) * CLOCKS_PER_SEC);
        t=clock();
        counter = 0;
    }

    // Increment counter
    ++counter;

    // Get the figure as a string
    sprintf(fps_text,"FPS: %.2f",fps);

    // Draw the string on the image
    //cvPutText(img, fps_text, cv::Point(0, 15), &font, cv::Scalar(255,255,255));
    cv::putText(img, fps_text, cv::Point(0, 25), cv::FONT_HERSHEY_SIMPLEX | cv::FONT_ITALIC, 1.0, cv::Scalar(255, 255, 255));

}

//! Draw the original video frame and the stabilized frame.  
//  Also draw a combined frame showing vectors from original to previous
int drawImages( cv::Mat& origImg,  cv::Mat& newImg, 
               std::vector<distPoint>* distancePoints, float shakeThreshold,
               float** distTable) 
{  
    int rows = origImg.rows;
    int cols = origImg.cols;

    // Show the original (reference) image
    cv::imshow("Orig Frame", origImg);
    cv::moveWindow("Orig Frame", 0, 0);

    // Show the latest image from webcam
    cv::imshow("New Frame", newImg);
    cv::moveWindow("New Frame", 0, rows);

    // Coordinates of the image center (used when rotating the image)
    cv::Point2d img_center;  
    img_center.x = (float)cols/2.0f;
    img_center.y = (float)rows/2.0f;

    std::vector<distPoint> usefulDistPoints;

    // Only use useful distancePoints for the calculations 
    for(unsigned int i=0; i<distancePoints->size(); i++) {
        
        distPoint dp = distancePoints->at(i);

        // Don't do anything with too large of differences
        if(euclideanDist(dp) > shakeThreshold) continue;
        // Don't do anything if these are duplicate points
        if(dp.dup) continue;
        // Don't do anything if the points don't match 
        if(!dp.match) continue;

        usefulDistPoints.push_back(dp);
    }

    // TODO Highlight which points were matched
    // If no interesting points are found, the code cannot
    // continue
    if(usefulDistPoints.size() <= 1) {
        return 0;
    }

    //----------------------------------------------
    // Rotate the image
    //----------------------------------------------
    float rotation;
    rotation = getRotation(usefulDistPoints);

    // Create the rotated image
    //cv::Mat rotatedImg = cv::(newImg);
    cv::Mat rotatedImg = newImg;

    // Create a matrix describing the rotation
    //cv::Mat rotMtx = cv::Mat(2, 3, CV_32FC1);
    auto rotMtx = cv::getRotationMatrix2D_(img_center, rotation, 1.0f);
    //Cv2DRotationMatrix(img_center, rotation, 1.0f, rotMtx);
    // Rotate the matrix
    cv::warpAffine(newImg, rotatedImg, rotMtx, newImg.size());
    //cvWarpAffine(newImg, rotatedImg, rotMtx,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, cv::All(255));
    //cvReleaseMat(&rotMtx);

    //----------------------------------------------
    // Scale the image
    //----------------------------------------------
    float scale;
    scale = getScalingFactor(usefulDistPoints, distTable, rotation,img_center);

    // Create the scaled image
    int scaledCols = (int)(((float)origImg.cols)/scale);
    int scaledRows = (int)(((float)origImg.rows)/scale);
    cv::Mat scaledImg = cv::Mat(cv::Size(scaledCols, scaledRows), 
        origImg.depth(), origImg.channels());

    // Create a matrix describing the scale (don't rotate)
    cv::resize(rotatedImg, scaledImg,cv::Size(scale,scale));

    //----------------------------------------------
    // Translate (shift) the image
    //----------------------------------------------
    cv::Point2d shift;
    shift = getTranslation(usefulDistPoints, scale, rotation, img_center);

    int srcX, srcY, dstX, dstY, rectWidth, rectHeight;
    // Shift the image
    if(-1*shift.x >= 0) {
        // New frame moved right, need to move it back to the left
        srcX = abs((int)shift.x);
        dstX = 0;
    }
    else {
        // New frame moved left, need to move it back to the right
        srcX = 0;
        dstX = abs((int)shift.x);
    }
    if(-1*shift.y >= 0) {
        // New frame moved down, need to move it back up
        srcY = abs((int)shift.y);
        dstY = 0;
    }
    else {
        // New frame moved up, need to move it back down
        srcY = 0;
        dstY = abs((int)shift.y);
    }

    // Set the bounds for the part of the image that we want to copy
	
    rectWidth = (std::min)(cols-dstX-1,scaledCols-srcX-1);
    rectHeight = (std::min)(rows-dstY-1, scaledRows-srcY-1);

    // Sanity check the output dimensions
    if(dstX+rectWidth >= cols || dstY+rectHeight >= rows ||
        srcX+rectWidth >= scaledCols || srcY+rectHeight >= scaledRows) {
            printf("BAD DIMENSIONS!\n");
            exit(-1);
    }
    
    // Created the shifted (and rotated) image
    //cv::Mat shiftedImg = cvCloneImage(origImg);
    cv::Mat shiftedImg = origImg;
    uchar* data = (uchar *) shiftedImg.data;
    // Fill in the backgroup with white
    memset(data, 255, sizeof(uchar)*cols*rows*shiftedImg.channels());
    
    // Determine the region of the rotated image to copy
    cv::Rect srcROI = cv::Rect(srcX, srcY, rectWidth, rectHeight);
    cv::Rect dstROI = cv::Rect(dstX, dstY, rectWidth, rectHeight);
    // Set the region of interest bounds
    //cvSetImageROI(scaledImg, srcROI);
    //cvSetImageROI(shiftedImg, dstROI);
    scaledImg = scaledImg(srcROI);    
    shiftedImg = shiftedImg(dstROI);

    // Copy from the rotated image to the shifted image
    //cvCopy(scaledImg, shiftedImg);
    scaledImg.copyTo(shiftedImg);    
    
    // Reset the region of interest so we can view the entire image    
    //cvResetImageROI(shiftedImg);
    //cvResetImageROI(scaledImg);
    
    //-------------------------------------------
    // Display the images
    //-------------------------------------------
    
    // Draw the FPS figure
    drawFPS(shiftedImg);

    // Shifted (final) image
    cv::imshow("Output Frame", shiftedImg);
    cv::moveWindow("Output Frame", cols, 0);

    // Uncomment the following lines to see intermediate images
    
    // Rotated image
    cv::imshow("Rotated Frame", rotatedImg);
    cv::moveWindow("Rotated Frame", cols, rows);
 
    // Scaled image
    cv::imshow("Scaled Frame", scaledImg);
    cv::moveWindow("Scaled Frame", 0, 2*rows);
    
    /*cvReleaseImage(&shiftedImg);
    cvReleaseImage(&rotatedImg);
    cvReleaseImage(&scaledImg);*/

    return 0;
}

//! Draw a single feature on the image
void drawIpoint( cv::Mat& img, Ipoint &ipt)
{
    float s, o;
    int r1, c1, r2, c2, lap;

    s = ((9.0f/1.2f) * ipt.scale) / 3.0f;
    o = ipt.orientation;
    lap = ipt.laplacian;
    r1 = fRound(ipt.y);
    c1 = fRound(ipt.x);

    // Green line indicates orientation
    if (o) // Green line indicates orientation
    {
        c2 = fRound(s * cos(o)) + c1;
        r2 = fRound(s * sin(o)) + r1;        
        cv::line(img, cv::Point(c1, r1), cv::Point(c2, r2), cv::Scalar(0, 255, 0));
    }
    else  // Green dot if using upright version
        cv::circle(img, cv::Point(c1,r1), 1, cv::Scalar(0, 255, 0),-1);

    if (lap >= 0)
    { // Blue circles indicate light blobs on dark backgrounds
        cv::circle(img, cv::Point(c1,r1), fRound(s), cv::Scalar(255, 0, 0),1);
    }
    else
    { // Red circles indicate light blobs on dark backgrounds
        cv::circle(img, cv::Point(c1,r1), fRound(s), cv::Scalar(0, 0, 255),1);
    }
}

//! Draw all the Ipoints in the provided vector
void drawIpoints( cv::Mat& img, std::vector<Ipoint> &ipts)
{
    Ipoint *ipt;
    float s, o;
    int r1, c1, r2, c2, lap;

    for(unsigned int i = 0; i < ipts.size(); i++)
    {
        ipt = &ipts.at(i);
        s = ((9.0f/1.2f) * ipt->scale) / 3.0f;
        o = ipt->orientation;
        lap = ipt->laplacian;
        r1 = fRound(ipt->y);
        c1 = fRound(ipt->x);
        c2 = fRound(s * cos(o)) + c1;
        r2 = fRound(s * sin(o)) + r1;

        if (o) // Green line indicates orientation
            cv::line(img, cv::Point(c1, r1), cv::Point(c2, r2), cv::Scalar(0, 255, 0));
        else  // Green dot if using upright version
            cv::circle(img, cv::Point(c1,r1), 1, cv::Scalar(0, 255, 0),-1);

        if (lap == 1)
        { // Blue circles indicate light blobs on dark backgrounds
            cv::circle(img, cv::Point(c1,r1), fRound(s), cv::Scalar(255, 0, 0),1);
        }
        else
        { // Red circles indicate light blobs on dark backgrounds
            cv::circle(img, cv::Point(c1,r1), fRound(s), cv::Scalar(0, 0, 255),1);
        }

    }
}

//! Simply draw the features as small red dots
void drawIpointsAsDots( cv::Mat& img, std::vector<Ipoint> &ipts)
{
    Ipoint *ipt;
    float s, o;
    int r1, c1, r2, c2, lap;

    for(unsigned int i = 0; i < ipts.size(); i++)
    {
        ipt = &ipts.at(i);
        //s = ((9.0f/1.2f) * ipt->scale) / 3.0f;
        s = 5;
        o = ipt->orientation;
        lap = ipt->laplacian;
        r1 = fRound(ipt->y);
        c1 = fRound(ipt->x);
        c2 = fRound(s * cos(o)) + c1;
        r2 = fRound(s * sin(o)) + r1;

        // Draw a red dot
        cv::circle(img, cv::Point(c1,r1), fRound(s), cv::Scalar(0, 0, 255),0);
    }
}

//! Draw a single feature on the image
void drawPoint( cv::Mat& img, Ipoint &ipt)
{
    float s, o;
    int r1, c1;

    s = 3;
    o = ipt.orientation;
    r1 = fRound(ipt.y);
    c1 = fRound(ipt.x);

    cv::circle(img, cv::Point(c1,r1), fRound(s), COLOURS[ipt.clusterIndex%NCOLOURS], -1);
    cv::circle(img, cv::Point(c1,r1), fRound(s+1), COLOURS[(ipt.clusterIndex+1)%NCOLOURS], 2);

}

//! Draw all features on the image
void drawPoints( cv::Mat& img, std::vector<Ipoint> &ipts)
{
    float s, o;
    int r1, c1;

    for(unsigned int i = 0; i < ipts.size(); i++)
    {
        s = 3;
        o = ipts[i].orientation;
        r1 = fRound(ipts[i].y);
        c1 = fRound(ipts[i].x);

        cv::circle(img, cv::Point(c1,r1), fRound(s), COLOURS[ipts[i].clusterIndex%NCOLOURS], -1);
        cv::circle(img, cv::Point(c1,r1), fRound(s+1), COLOURS[(ipts[i].clusterIndex+1)%NCOLOURS], 2);
    }
}

//! Draw descriptor windows around Ipoints in the provided vector
void drawWindows( cv::Mat& img, std::vector<Ipoint> &ipts)
{
    Ipoint *ipt;
    float s, o, cd, sd;
    int x, y;
    cv::Point2d src[4];

    for(unsigned int i = 0; i < ipts.size(); i++)
    {
        ipt = &ipts.at(i);
        s = (10 * ipt->scale);
        o = ipt->orientation;
        y = fRound(ipt->y);
        x = fRound(ipt->x);
        cd = cos(o);
        sd = sin(o);

        src[0].x=sd*s+cd*s+x;   src[0].y=-cd*s+sd*s+y;
        src[1].x=sd*s+cd*-s+x;  src[1].y=-cd*s+sd*-s+y;
        src[2].x=sd*-s+cd*-s+x; src[2].y=-cd*-s+sd*-s+y;
        src[3].x=sd*-s+cd*s+x;  src[3].y=-cd*-s+sd*s+y;

        if (o) // Draw orientation line
            cv::line(img, cv::Point(x, y),
            cv::Point(fRound(s*cd + x), fRound(s*sd + y)), cv::Scalar(0, 255, 0),1);
        else  // Green dot if using upright version
            cv::circle(img, cv::Point(x,y), 1, cv::Scalar(0, 255, 0),-1);


        // Draw box window around the point
        cv::line(img, cv::Point(fRound(src[0].x), fRound(src[0].y)),
            cv::Point(fRound(src[1].x), fRound(src[1].y)), cv::Scalar(255, 0, 0),2);
        cv::line(img, cv::Point(fRound(src[1].x), fRound(src[1].y)),
            cv::Point(fRound(src[2].x), fRound(src[2].y)), cv::Scalar(255, 0, 0),2);
        cv::line(img, cv::Point(fRound(src[2].x), fRound(src[2].y)),
            cv::Point(fRound(src[3].x), fRound(src[3].y)), cv::Scalar(255, 0, 0),2);
        cv::line(img, cv::Point(fRound(src[3].x), fRound(src[3].y)),
            cv::Point(fRound(src[0].x), fRound(src[0].y)), cv::Scalar(255, 0, 0),2);

    }
}

//! Euclidian distance from one coordinate to a second
float euclideanDist(distPoint p) {

    return sqrt((p.x2-p.x1)*(p.x2-p.x1)+(p.y2-p.y1)*(p.y2-p.y1));
}

//! Euclidian distance between two pairs of coordinates
float euclideanDist(float x1, float y1, float x2, float y2) {

    return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

//! Release the memory allocated for the distance table
void freeDistanceTable(float** distTable, unsigned int numPts) 
{
    for(unsigned int i = 0; i < numPts; i++) 
    {
        free(distTable[i]);
    }
    free(distTable);
}

//! Convert image from 4 channel color to 4 channel grayscale
 cv::Mat getGray(const  cv::Mat& img)
{
    // Check we have been supplied a non-null img pointer
    if (img.empty()) 
    {
        printf("Unable to create grayscale image.  No image supplied");
        exit(-1);
    }

    cv::Mat gray8;
    cv::Mat gray32;

    // Allocate space for the grayscale
    gray32 = cv::Mat(img.rows,img.cols, CV_32FC1, 1);
    
    if (img.channels() == 1) {
        gray8 = img;        
    }
    else 
    {
        gray8 = cv::Mat(img.rows,img.cols, CV_8UC1, 1);
        cv::cvtColor(img, gray8, cv::COLOR_BGR2GRAY);        
    }
    
    gray8.convertTo(gray32, CV_32FC1, 1.0 / 255.0, 0);
    
    return gray32;
}

//! Look at the valid points and determine the median orientation
//  difference with the original image
//  TODO Remove some outliers?
float getRotation(std::vector<distPoint> distancePoints) 
{
    // Create a vector of all of the orientation differences
    std::vector<float> orientations;
    for(unsigned int i = 0; i < distancePoints.size(); i++) {
        
        distPoint dp = distancePoints.at(i);

        float adjustedOrientation = dp.orientationDiff;
        // Convert the angle to the range -180:180.  The loops may
        // be overkill, but should be guaranteed to work for any value
        while(adjustedOrientation <= -CV_PI) {
            adjustedOrientation += (float)(2*CV_PI);
        }
        while(adjustedOrientation > CV_PI) {
            adjustedOrientation -= (float)(2*CV_PI);
        }

        orientations.push_back(dp.orientationDiff);
    }

    // Determine the median orientation difference
    float medianOrientation = median(orientations);

    // Convert to degrees
    float medianOrientationDegrees = medianOrientation*180.0f/(float)CV_PI;

    return medianOrientationDegrees;
}

//! Determine how much the image is scaled by comparing the vector
//  from the center of the image to each point, then make it match
//  the original point
float getScalingFactor(std::vector<distPoint> distancePoints, 
                       float** distTable, float rotation,
                       cv::Point2d imgCenter) 
{
    std::vector<float> scales;
    int numDistPoints = (int)distancePoints.size();

    // Compare each distance point to all previous distance
    // points.  For each point, determine the ratio between
    // these points and their reference counterparts (this
    // will tell us the scaling factor
    for(int i=0; i< numDistPoints; i++) {
        
        distPoint dp = distancePoints.at(i);

        // Rotate each distancePoint around the image center by 'rotation'
        // degrees
        float x1 = dp.x1-imgCenter.x;
        float y1 = dp.y1-imgCenter.y;

        // Convert degrees to radians
        float theta1 = -1.0f*rotation*(float)(CV_PI/180.0);

        // x' = x cos(theta) - y sin(theta)
        float x1prime = x1*cosf(theta1) - 
                        y1*sinf(theta1);
        // y' = x sin(theta) + y cos(theta)
        float y1prime = x1*sinf(theta1) + 
                        y1*cosf(theta1);

        // Need to add the new points back to the image center
        x1prime += imgCenter.x;
        y1prime += imgCenter.y;

        for(int j = i-1; j >= 0; j--) {

            distPoint other = distancePoints.at(j);

            // Rotate each distancePoint around the image center by 'rotation'
            // degrees
            float x2 = other.x1-imgCenter.x;
            float y2 = other.y1-imgCenter.y;

            // Convert degrees to radians
            float theta2 = -1.0f*rotation*(float)(CV_PI/180.0);

            // x' = x cos(theta) - y sin(theta)
            float x2prime = x2*cosf(theta2) - 
                            y2*sinf(theta2);
            // y' = x sin(theta) + y cos(theta)
            float y2prime = x2*sinf(theta2) + 
                            y2*cosf(theta2);

            // Need to add the new points back to the image center
            x2prime += imgCenter.x;
            y2prime += imgCenter.y;

            float oldDist = distTable[dp.point2][other.point2];
            float newDist = euclideanDist(x1prime, y1prime, x2prime, y2prime);
            scales.push_back(newDist/oldDist);
        }
    }
    
    // Get the median scale value
    float scale = median(scales);

    return scale;
}

//! Rotate the points and scale them (as will be done for the entire
//  image), then determine how they should be translated
cv::Point2d getTranslation(std::vector<distPoint> distancePoints,
                            float scale, float rotation, 
                            cv::Point2d imgCenter) 
{
    std::vector<float> shiftX;
    std::vector<float> shiftY;
    cv::Point2d shiftCoords;

    int numPoints = distancePoints.size();
    for(int i = 0; i < numPoints; i++) 
    {

        // Rotate each distancePoint around the image center by 'rotation'
        // degrees
        float x1 = distancePoints.at(i).x1-imgCenter.x;
        float y1 = distancePoints.at(i).y1-imgCenter.y;

        // Convert degrees to radians
        float theta = -1.0f*rotation*(float)(CV_PI/180.0);

        // x' = x cos(theta) - y sin(theta)
        float x1prime = x1*cosf(theta) - 
                        y1*sinf(theta);
        // y' = x sin(theta) + y cos(theta)
        float y1prime = x1*sinf(theta) + 
                        y1*cosf(theta);

        // Need to add the new points back to the image center
        x1prime += imgCenter.x;
        y1prime += imgCenter.y;

        // Divide all distances by the scaling factor--this will
        // scale the image relative to the point (0,0)
        float newDist = euclideanDist(0, 0, x1prime, y1prime)/scale;

        // Compute the angle from (0,0) to the rotated coordinates
        float angle = atanf(y1prime/x1prime);
        // The new X coord is found with cos(angle) = adj/hyp 
        // hyp == newDist, angle == angle
        x1prime = cosf(angle)*newDist;
        // The new Y coord is found with sin(angle) = opp/hyp
        y1prime = sinf(angle)*newDist;

        // The points should now be the right orientation and size,
        // and a translation should put them on top of the reference points
        shiftX.push_back(distancePoints.at(i).x2-x1prime);
        shiftY.push_back(distancePoints.at(i).y2-y1prime);
    }

    shiftCoords.x = median(shiftX);
    shiftCoords.y = median(shiftY);

    return shiftCoords;
}

//! Load the SURF features from file
void loadSurf(char *filename, std::vector<Ipoint> &ipts)
{
    int descriptorLength, count;
    std::ifstream infile(filename);

    // clear the ipts vector first
    ipts.clear();

    // read descriptor length/number of ipoints
    infile >> descriptorLength;
    infile >> count;

    // for each ipoint
    for (int i = 0; i < count; i++)
    {
        Ipoint ipt;

        // read vals
        infile >> ipt.scale;
        infile >> ipt.x;
        infile >> ipt.y;
        infile >> ipt.orientation;
        infile >> ipt.laplacian;
        infile >> ipt.scale;

        // read descriptor components
        for (int j = 0; j < 64; j++)
            infile >> ipt.descriptor[j];

        ipts.push_back(ipt);

    }
}

//! Returns the median value for a vector of floats
float median(std::vector<float> v) {

    int size = v.size();
    if(size == 0) {
        printf("No elements for median\n");
        exit(-1);
    }

    std::sort(v.begin(), v.end());

    return v.at(size/2);
}


IpVec* mergeIpts(IpVec latestPts, IpVec prevPts, std::vector<distPoint> &distPoints) 
{

    IpVec* mergedIpts = new IpVec(prevPts);
    std::vector<int> deleteList;

    // Point2 of distance points correspond to the main list of points,
    // point1 corresponds to the latest points
    for(unsigned int i = 0; i < distPoints.size(); i++) {
        if(distPoints.at(i).dup == 1 || distPoints.at(i).match == 1) {
            deleteList.push_back(distPoints.at(i).point1);
        }
    }
    std::sort(deleteList.begin(), deleteList.end());

    for(int i = deleteList.size()-1; i >= 0; i--) {
        latestPts.erase(latestPts.begin()+deleteList.at(i));
    }

    mergedIpts->insert(mergedIpts->end(), latestPts.begin(), latestPts.end());

    return mergedIpts;
}

//! Save the SURF-ified image with a -surf.jpg extension
void saveImage(const  cv::Mat& img, char *originalName)
{
    char *newName = NULL;

    int len = (int)strlen(originalName);

    // strip any extension off of original name
    for(int i = len-1; i >= 0; i--){
        if(originalName[i]=='.') {
            originalName[i]='\0';
            break;
        }
    }

    // create new name
    size_t newNameSize = (strlen(originalName)+strlen("-surf.jpg")+1)*sizeof(char);
    newName = (char *)alloc(newNameSize);

    strcpy(newName, originalName);
    strcat(newName, "-surf.jpg");
    newName[newNameSize-1] = '\0';

    int params[3];
    params[0] = cv::IMWRITE_JPEG_QUALITY;
    params[1] = 100;
    params[2] = 0;

    // save the new file
    std::vector<int> vparams = { params[0],params[1],params[2] };
    cv::imwrite(newName, img, vparams);
    //cvSaveImage(newName, img, params);

    free(newName);
}

//! Show the provided image and wait for keypress
void showImage(const  cv::Mat& img, bool blocking)
{
    cv::namedWindow("Surf", cv::WINDOW_AUTOSIZE);
    cv::imshow("Surf", img);
    if(blocking)     
        cv::waitKey(0);    
}


//! Show the provided image in titled window and wait for keypress
void showImage(char *title, const  cv::Mat& img, bool blocking)
{
    cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
    cv::imshow(title, img);

    if(blocking) {
        cv::waitKey(0);
    }
}

//! Write Text on Image
void writeImageName( cv::Mat& img, char * s)
{
    //cv::Font font;
    //cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 1.0,1.0,0,2);
    //cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 0.5,0.5,0,1);
    //cvPutText (img,s,cv::Point(10,static_cast<int>(0.9*(img->height))), &font, cv::Scalar(255,255,0));
    cv::putText(img,s, cv::Point(10, static_cast<int>(0.9 * (img.rows))), cv::FONT_HERSHEY_SIMPLEX | cv::FONT_ITALIC, 0.5, cv::Scalar(255, 255, 0),  0.5, 0, 1);
    
}

//! Write Text on Image
void writeImageText( cv::Mat& img, char * s)
{
    //cv::Font font;
    //cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 1.0,1.0,0,2);
    //cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 0.5,0.5,0,1);
    //cvPutText (img,s,cv::Point(10,10), &font, cv::Scalar(255,255,0));
    cv::putText(img, s, cv::Point(10, 10), cv::FONT_HERSHEY_SIMPLEX | cv::FONT_ITALIC, 0.5, cv::Scalar(255, 255, 0), 0.5, 0, 1);
}

//! Save the SURF features to file
void writeIptsToFile(char *path, std::vector<Ipoint> &ipts)
{

    char* fullpath = smartStrcat(path, "/SurfIpts.log");
    std::ofstream outfile(fullpath);

    // output descriptor length
    outfile << "64\n";
    outfile << ipts.size() << "\n";

    // create output line as:  scale  x  y  des
    for(unsigned int i=0; i < ipts.size(); i++)
    {

        outfile << ipts.at(i).scale << "  ";
        outfile << ipts.at(i).x << " ";
        outfile << ipts.at(i).y << " ";
        outfile << ipts.at(i).orientation << " ";
        outfile << ipts.at(i).laplacian << " ";
        outfile << ipts.at(i).scale << " ";
        for(int j=0; j<64; j++) {
            outfile << ipts.at(i).descriptor[j] << " ";
        }
        outfile << "\n";

    }

    outfile.close();

    free(fullpath);
}

void writeDptsToFile(char *path, std::vector<distPoint> &dpts)
{

    char* fullpath = smartStrcat(path, "/SurfDpts.log");
    std::ofstream outfile(fullpath);

    outfile << "num points: " << dpts.size() << "\n";

    // create output line as:  scale  x  y  des
    for(unsigned int i=0; i < dpts.size(); i++)
    {

        outfile << "point1: " << dpts.at(i).point1 << "\n";
        outfile << "x1: " << dpts.at(i).x1 << "\n";
        outfile << "y1: " << dpts.at(i).y1 << "\n";
        outfile << "point2: " << dpts.at(i).point2 << "\n";
        outfile << "x2: " << dpts.at(i).x2 << "\n";
        outfile << "y2: " << dpts.at(i).y2 << "\n";
        outfile << "dist: " << dpts.at(i).dist << "\n";
        outfile << "orientation: " << dpts.at(i).orientationDiff << "\n";
        outfile << "scale: " << dpts.at(i).scaleDiff << "\n";
        if(dpts.at(i).dup) {
            outfile << "dup\n";
        }
        else {
            outfile << "not dup\n";
        }
        if(dpts.at(i).match) {
            outfile << "match\n";
        }
        else {
            outfile << "not match\n";
        }
        outfile << "\n";

    }
    outfile.close();

    free(fullpath);
}
