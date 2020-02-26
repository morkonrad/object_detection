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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <CL/cl.h>
#include <time.h>
#include <algorithm>

#ifdef _WIN32
// Required for gethostname
#include <winsock2.h>
#endif

#include "eventlist.h"
#include "utils.h"

//! Constructor
EventList::EventList()
{

}


//! Destructor
EventList::~EventList()
{
    // TODO Changes these loops to use iterators

    // Release kernel events
    for(int i = 0; i < (int)kernel_events.size(); i++) {
        clReleaseEvent(this->kernel_events[i].first);
        free(this->kernel_events[i].second);
    }

    // Release IO events
    for(int i = 0; i < (int)io_events.size(); i++) {
        clReleaseEvent(this->io_events[i].first);
        free(this->io_events[i].second);
    }

    // Compile events and User events use static char*s, no need to free them

    // Free the event lists
    this->kernel_events.clear();
    this->io_events.clear();
}


char* EventList::createFilenameWithTimestamp()
{
    // TODO Make this nicer
    int maxStringLen = 100;
    char* timeStr = NULL;
    timeStr = (char*)alloc(sizeof(char)*maxStringLen);

    time_t rawtime;
    struct tm* timeStruct;

    time(&rawtime);
    timeStruct = localtime(&rawtime);

    strftime(timeStr, maxStringLen, "/Events_%Y_%m_%d_%H_%M_%S.surflog", timeStruct);

    return timeStr;
}

//! Dump a CSV file with event information
void EventList::dumpCSV(char* path)
{

    char* fullpath = NULL;
    FILE* fp =  NULL;

    // Construct a filename based on the current time
    char* filename = this->createFilenameWithTimestamp();
    fullpath = smartStrcat(path, filename);

    // Try to open the file for writing
    fp = fopen(fullpath, "w");
    if(fp == NULL) {
        printf("Error opening %s\n", fullpath);
        exit(-1);
    }

    // Write some information out about the environment

    char* tmp;
    char* tmp2;

    // Write the device name
    tmp = cl_getDeviceName();
    if(isUsingImages()) {
        tmp2 = smartStrcat(tmp, " (images)");
    }
    else {
        tmp2 = smartStrcat(tmp, " (buffers)");
    }
    fprintf(fp, "Info;\t%s\n", tmp2);
    free(tmp);
    free(tmp2);

    // Write the vendor name
    tmp = cl_getDeviceVendor();
    fprintf(fp, "Info;\t%s\n", tmp);
    free(tmp);

    // Write the driver version
    tmp = cl_getDeviceDriverVersion();
    fprintf(fp, "Info;\tDriver version %s\n", tmp);
    free(tmp);

    // Write the device version
    tmp = cl_getDeviceVersion();
    fprintf(fp, "Info;\t%s\n", tmp);
    free(tmp);

    // Write the hostname
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2,2), &wsaData);
#endif
    char hostname[50];
    if(gethostname(hostname, 50) != 0) {
        printf("Error getting hostname\n");
    }
    else {
        fprintf(fp, "Info;\tHost %s\n", hostname);
    }

    int kernelEventSize = this->kernel_events.size();
    for(int i = 0; i < kernelEventSize; i++) {
        fprintf(fp, "Kernel;\t%s;\t%3.3f\n", this->kernel_events[i].second,
            cl_computeExecTime(this->kernel_events[i].first));
    }
    int ioEventSize = this->io_events.size();
    for(int i = 0; i < ioEventSize; i++) {
        fprintf(fp, "IO;\t%s;\t%3.3f\n", this->io_events[i].second,
            cl_computeExecTime(this->io_events[i].first));
    }
    int compileEventSize = this->compile_events.size();
    for(int i = 0; i < compileEventSize; i++) {
        fprintf(fp, "Compile;\t%s;\t%3.3f\n", this->compile_events[i].second,
            compile_events[i].first);
    }
    int userEventSize = this->user_events.size();
    for(int i = 0; i < userEventSize; i++) {
        fprintf(fp, "User;\t%s;\t%3.3f\n", this->user_events[i].second,
            user_events[i].first);
    }

    fclose(fp);

    free(filename);
    free(fullpath);
}

//! Add a new compile event
void EventList::newCompileEvent(double time, char* desc)
{
    time_tuple tuple;

    tuple.first = time;
    tuple.second = desc;

    this->compile_events.push_back(tuple);
}


//! Add a new kernel event
void EventList::newKernelEvent(cl_event event, char* desc)
{
    event_tuple tuple;

    tuple.first = event;
    tuple.second = desc;

    this->kernel_events.push_back(tuple);
}


//! Add a new IO event
void EventList::newIOEvent(cl_event event, char* desc)
{
    event_tuple tuple;

    tuple.first = event;
    tuple.second = desc;

    this->io_events.push_back(tuple);
}

//! Add a new user event
void EventList::newUserEvent(double time, char* desc)
{
    time_tuple tuple;

    tuple.first = time;
    tuple.second = desc;

    this->user_events.push_back(tuple);
}

//! Print event information for all events
void EventList::printAllEvents()
{
    this->printCompileEvents();
    this->printKernelEvents();
    this->printIOEvents();
    this->printUserEvents();
}


//! Print event information for all entries in compile_events vector
void EventList::printCompileEvents()
{

    int numEvents = this->compile_events.size();

    for(int i = 0; i < numEvents; i++)
    {
        printf("Compile Event %d: %s\n", i, this->compile_events[i].second);
        printf("\tDURATION: %f\n", this->compile_events[i].first);
    }
}


//! Print event information for all entries in io_events vector
void EventList::printIOEvents()
{

    int numEvents = this->io_events.size();
    cl_int status;
    cl_ulong timer;

    for(int i = 0; i < numEvents; i++)
    {
        printf("Kernel Event %d: %s\n", i, this->io_events[i].second);

        status = clGetEventProfilingInfo(this->io_events[i].first,
            CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        printf("\tENQUEUE: %lu\n", timer);

        status = clGetEventProfilingInfo(this->io_events[i].first,
            CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        printf("\tSUBMIT:  %lu\n", timer);

        status = clGetEventProfilingInfo(this->io_events[i].first,
            CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        printf("\tSTART:   %lu\n", timer);

        status = clGetEventProfilingInfo(this->io_events[i].first,
            CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        printf("\tEND:     %lu\n", timer);
    }
}


//! Print event information for all entries in kernel_events vector
void EventList::printKernelEvents()
{

    int numEvents = this->kernel_events.size();
    cl_int status;
    cl_ulong timer;

    for(int i = 0; i < numEvents; i++)
    {

        printf("Kernel event %d: %s\n", i, kernel_events[i].second);

        status = clGetEventProfilingInfo(this->kernel_events[i].first,
            CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        printf("\tENQUEUE: %lu\n", timer);

        status = clGetEventProfilingInfo(this->kernel_events[i].first,
            CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        printf("\tSUBMIT:  %lu\n", timer);

        status = clGetEventProfilingInfo(this->kernel_events[i].first,
            CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        printf("\tSTART:   %lu\n", timer);

        status = clGetEventProfilingInfo(this->kernel_events[i].first,
            CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        printf("\tEND:     %lu\n", timer);
    }
}

//! Print event information for all entries in user_events vector
void EventList::printUserEvents()
{

    int numEvents = this->user_events.size();

    for(int i = 0; i < numEvents; i++)
    {
        printf("User Event %d: %s\n", i, this->user_events[i].second);
        printf("\tDURATION: %f\n", this->user_events[i].first);
    }
}

//! Print execution times of all events
void EventList::printAllExecTimes()
{
    this->printCompileExecTimes();
    this->printKernelExecTimes();
    this->printIOExecTimes();
    this->printUserExecTimes();
}

//! Print execution times for all entries in compile_events vector
void EventList::printCompileExecTimes()
{

    int numEvents = this->compile_events.size();

    for(int i = 0; i < numEvents; i++)
    {
        printf("Compile: %f: %s\n", this->compile_events[i].first,
            this->compile_events[i].second);
    }
}


//! Print execution times for all entries in io_events vector
void EventList::printIOExecTimes()
{

    int numEvents = this->io_events.size();

    for(int i = 0; i < numEvents; i++)
    {
        printf("IO:      %3.3f: %s\n", cl_computeExecTime(this->io_events[i].first),
            this->io_events[i].second);
    }
}

//! Print execution times for all entries in kernel_events vector
void EventList::printKernelExecTimes()
{

    int numEvents = this->kernel_events.size();

    for(int i = 0; i < numEvents; i++)
    {
        printf("Kernel:  %3.3f: %s\n", cl_computeExecTime(this->kernel_events[i].first),
            this->kernel_events[i].second);
    }
}

//! Print execution times for all entries in user_events vector
void EventList::printUserExecTimes()
{

    int numEvents = this->user_events.size();

    for(int i = 0; i < numEvents; i++)
    {
        printf("User: %f: %s\n", this->user_events[i].first,
            this->user_events[i].second);
    }
}






//! Dump a CSV file with event information
void EventList::dumpTraceCSV(char* path)
{

    char* fullpath = NULL;
    FILE* fp =  NULL;

    // Construct a filename based on the current time
    char* filename = this->createFilenameWithTimestamp();
    fullpath = smartStrcat(path, filename);

    // Try to open the file for writing
    fp = fopen(fullpath, "w");
    if(fp == NULL) {
        printf("Error opening %s\n", fullpath);
        exit(-1);
    }

    // Write some information out about the environment

    char* tmp;
    char* tmp2;

    // Write the device name
    tmp = cl_getDeviceName();
    if(isUsingImages()) {
        tmp2 = smartStrcat(tmp, " (images)");
    }
    else {
        tmp2 = smartStrcat(tmp, " (buffers)");
    }
    fprintf(fp, "Info;\t%s\n", tmp2);
    free(tmp);
    free(tmp2);

    // Write the vendor name
    tmp = cl_getDeviceVendor();
    fprintf(fp, "Info;\t%s\n", tmp);
    free(tmp);

    // Write the driver version
    tmp = cl_getDeviceDriverVersion();
    fprintf(fp, "Info;\tDriver version %s\n", tmp);
    free(tmp);

    // Write the device version
    tmp = cl_getDeviceVersion();
    fprintf(fp, "Info;\t%s\n", tmp);
    free(tmp);

    // Write the hostname
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2,2), &wsaData);
#endif
    char hostname[50];
    if(gethostname(hostname, 50) != 0) {
        printf("Error getting hostname\n");
    }
    else {
        fprintf(fp, "Info;\tHost %s\n", hostname);
    }

    int kernelEventSize = this->kernel_events.size();
    cl_ulong timer;
    cl_int status;

    for(int i = 0; i < kernelEventSize; i++)
    {

        fprintf(fp, "Kernel\t%s", this->kernel_events[i].second);
        //    cl_computeExecTime(this->kernel_events[i].first));

        status = clGetEventProfilingInfo(this->kernel_events[i].first,
            CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        fprintf(fp, "\t%lu", timer);

        status = clGetEventProfilingInfo(this->kernel_events[i].first,
            CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        fprintf(fp, "\t%lu", timer);

        status = clGetEventProfilingInfo(this->kernel_events[i].first,
            CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        fprintf(fp, "\t%lu", timer);

        status = clGetEventProfilingInfo(this->kernel_events[i].first,
            CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        fprintf(fp, "\t%lu\n", timer);


    }
    int ioEventSize = this->io_events.size();
    for(int i = 0; i < ioEventSize; i++)
    {
        fprintf(fp, "IO\t%s", this->io_events[i].second);
            //,cl_computeExecTime(this->io_events[i].first));

        status = clGetEventProfilingInfo(this->io_events[i].first,
            CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        fprintf(fp, "\t%lu", timer);

        status = clGetEventProfilingInfo(this->io_events[i].first,
            CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        fprintf(fp, "\t%lu", timer);

        status = clGetEventProfilingInfo(this->io_events[i].first,
            CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        fprintf(fp, "\t%lu", timer);

        status = clGetEventProfilingInfo(this->io_events[i].first,
            CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &timer, NULL);
        cl_errChk(status, "profiling", true);
        fprintf(fp, "\t%lu\n", timer);

    }

    /*
    int compileEventSize = this->compile_events.size();
    for(int i = 0; i < compileEventSize; i++)
    {
        fprintf(fp, "Compile;\t%s;\t%3.3f\n", this->compile_events[i].second,
            compile_events[i].first);
    }
    int userEventSize = this->user_events.size();
    for(int i = 0; i < userEventSize; i++) {
        fprintf(fp, "User;\t%s;\t%3.3f\n", this->user_events[i].second,
            user_events[i].first);
    }
    */
    fclose(fp);

    free(filename);
    free(fullpath);
}

