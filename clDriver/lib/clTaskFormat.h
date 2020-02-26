#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace txt_utils
{

struct if_not_prev_space
{
    // Is last encountered character space.
    bool m_is = false;

    bool operator()(const char c)
    {
        // Copy if last was not space, or current is not space.
        const bool ret = !m_is || c != ' ';
        m_is = c == ' ';
        return ret;
    }
};
/**
         * @brief remove_extra_whitespaces
         * Removes 2 or more whitespace
         * @param input
         * @param output
         */

inline void
remove_extra_whitespaces(const std::string &input, std::string &output)
{
    std::copy_if(std::begin(input), std::end(input), std::back_inserter(output), if_not_prev_space());
}

inline void
find_comments_block(const std::string& input,
                    const std::string& comment_begin,
                    const std::string& comment_end,
                    std::vector<size_t>& positions)
{
    auto pos = input.find(comment_begin);
    if (pos != std::string::npos)
    {
        positions.push_back(pos);
        pos = input.find(comment_end, pos + 1);
        positions.push_back(pos);
    }
}

inline void
remove_comments_rests(std::string& input)
{
    size_t pos = 0;
    while (pos != std::string::npos && !input.empty())
    {
        pos = input.find("*/");
        if (pos != std::string::npos)
            input.erase(pos, 2);
    }

}

inline void
remove_comments_block(std::string& input,
                      const std::string& comment_begin,
                      const std::string& comment_end)
{
    std::vector<size_t> positions{ 0,0 };
    while (positions.size() == 2 && !input.empty())
    {
        positions.clear();
        find_comments_block(input,
                            comment_begin,
                            comment_end,
                            positions);

        if (positions.size() < 2)
            break;

        if (positions.size() == 2)
        {
            const auto b = positions[0];
            const auto e = positions[1];
            //remove comment_block
            input.erase(b, e - b);
        }
    }
    remove_comments_rests(input);
}

inline void
remove_newline_tabs_carrieg(std::string& input)
{
    input.erase(std::remove_if(input.begin(), input.end(),
                               [](unsigned char x) {return x == '\n' || x == '\t' || x == '\r';}),
                input.end());
}

inline std::vector<size_t>
cut_sub_strings(const std::string& input,
                const std::string& separator)
{
    std::vector<size_t> positions;
    auto pos = input.find(separator);
    while (pos != std::string::npos)
    {
        positions.push_back(pos);
        pos = input.find(separator, pos + 1);
    }
    return positions;
}

/**
     * @brief cut_function_kernel
     * 1)Format task_src that possibly contains many kernel_functions.
     * 2)Search for function with name==func_name
     * @param task_src - string with function_kernels
     * @param func_name - name of function to find
     * @return if function with func_name found return function else ""
     */
inline std::string
cut_function_kernel(const std::string& task_src, const std::string& func_name)
{
    std::string task_formated;
    txt_utils::remove_extra_whitespaces(task_src, task_formated);
    txt_utils::remove_comments_block(task_formated, "/*", "*/");
    txt_utils::remove_comments_block(task_formated, "//", "\n");
    txt_utils::remove_newline_tabs_carrieg(task_formated);

    const auto positions = txt_utils::cut_sub_strings(task_formated, "kernel");
    if (positions.size() == 1)return task_formated;

    for (size_t id = 0;id < positions.size();id++)
    {
        const auto b = positions[id];
        size_t e = b;
        if (id < positions.size() - 1)
            e = positions[id + 1];
        else
            e = task_formated.size();

        const auto task = task_formated.substr(b, e - b);
        if (task.find(func_name) != std::string::npos)
            return task;
    }
    return "";
}

}

namespace rewrite
{
/**
* find all appearances of: __kernelvoid or kernelvoid and replace with:
* __kernel void or kernel void
*/
inline size_t
pre_process_kernel_func(std::string &source) {
    size_t beg = 0;
    std::string tag = "__kernelvoid";
    std::string formated = "__kernel void";
    beg = source.find(tag);
    if (beg == std::string::npos) {
        tag = "kernelvoid";
        formated = "kernel void";
        beg = source.find(tag);
    }

    if (beg != std::string::npos) {
        source.erase(beg, tag.length());
        source.insert(beg, formated.c_str(), formated.length());
    }
    return beg;
}

/**
* Search for the _kernel void or kernel void in the source.
* And cut from	the source a kernel function
*/
inline std::string
cut_kernel_function(std::string &source) 
{
    // Warning: 
    // This algorithm cuts text between apperance of kernel or __kernel 
    // and a next apperance of kernel or __kernel !!
   
    std::string before = "__kernel";
    std::string after = "__kernel";

    // search for __kernel or kernel tag
    auto beg = source.find(before);
    if (beg == std::string::npos) {
        before = "kernel";
        beg = source.find(before);
    }
    if (beg == std::string::npos) {
        source = "";
        return "";
    }

    auto end = source.find(after, beg + before.size());
    if (end == std::string::npos) {
        after = "kernel";
        end = source.find(after, beg + before.size());
    }

    std::string target;
    // if last kernel function than cut it off and reset the source
    if (beg < end && end == std::string::npos) {
        target = source.substr(beg, source.length() - beg);
        source = "";
    }
    else {
        if (beg >= end && end == std::string::npos) {
            source = "";
            return "";
        }

        target = source.substr(beg, end - beg);
        source = source.substr(end, source.length() - end);
    }

    return target;
}

inline std::string
insert_global_size_guard_and_args(
    const std::string& source,
    const std::array < std::string, 3 > & global_size_args)
{
    std::string target = source;
    std::stringstream global_size_assertion,global_size_arg_asserts;
    
    global_size_assertion << "if( get_global_id(0) >= " << global_size_args[0] << " )return;\n";
    global_size_assertion << "if( get_global_id(1) >= " << global_size_args[1] << " )return;\n";
    global_size_assertion << "if( get_global_id(2) >= " << global_size_args[2] << " )return;\n";

    global_size_arg_asserts 
        << "const int " << global_size_args[0] << " ,"
        << "const int " << global_size_args[1] << " ,"
        << "const int " << global_size_args[2] << " ,";

    auto find_and_insert = []( 
        const std::stringstream& asserts,
        const std::string token_after_kernel,
        const std::string input)->std::string
    {
        std::string before = "__kernel";
        std::string output = input;
        // search for __kernel or kernel token
        // and append/insert the asserts
        auto beg = output.find(before);
        if (beg == std::string::npos) {
            before = "kernel";
            beg = output.find(before);
        }
        if (beg == std::string::npos)
            return "";

        auto end = output.find(token_after_kernel, beg + before.size());
        if (end == std::string::npos)
            return "";

        output.insert(end + 1, asserts.str());

        return output;
    }; 

    //Find function call and a '(' bracket with args
    //Then append as a first args global_size_arg_asserts
    target = find_and_insert(global_size_arg_asserts, "(", target);
    
    //Find a kernel_function call and a '{' bracket with function body 
    //Then append as global_size_assertion
    target = find_and_insert(global_size_assertion, "{", target);
    
    return target;
}
/**
* search for the template: _kernel void ...{ or kernel void ...{
* and insert the global_size_assertion
*/
inline std::string
insert_global_size_guard(const std::string &source,
                         const std::array<size_t, 3> &global_sizes)
{
    std::string target = source;
    std::stringstream global_size_assertion;

    if (global_sizes[0] > 1)
        global_size_assertion << "if( get_global_id(0) > " << global_sizes[0] - 1
                              << " )return;\n";
    else
        global_size_assertion << "if( get_global_id(0) > " << global_sizes[0]
                              << " )return;\n";

    if (global_sizes[1] > 1)
        global_size_assertion << "if( get_global_id(1) > " << global_sizes[1] - 1
                              << " )return;\n";
    else
        global_size_assertion << "if( get_global_id(1) > " << global_sizes[1]
                              << " )return;\n";

    if (global_sizes[2] > 1)
        global_size_assertion << "if( get_global_id(2) > " << global_sizes[2] - 1
                              << " )return;\n";
    else
        global_size_assertion << "if( get_global_id(2) > " << global_sizes[2]
                              << " )return;\n";

    std::string before = "__kernel";
    std::string after = "{";

    // search for __kernel or kernel tag
    // and append/insert the global_size_assertion
    auto beg = target.find(before);
    if (beg == std::string::npos) {
        before = "kernel";
        beg = target.find(before);
    }
    if (beg == std::string::npos)
        return "";

    auto end = target.find(after, beg + before.size());
    if (end == std::string::npos)
        return "";

    target.insert(end + 1, global_size_assertion.str());

    return target;
}

inline bool
copy_include_headers(std::stringstream &out,
                                 const std::string &source)
{
    if (source.empty())
        return false;

    std::string query = "__kernel";
    // search for __kernel or kernel tag
    auto beg = source.find(query);
    if (beg == std::string::npos) {
        query = "kernel";
        beg = source.find(query);
    }
    if (beg == std::string::npos)
        return false;

    // copy from begin to the first appearance of query(__kernel or kernel)
	if (beg != 0)
	{
		const auto inch = source.substr(0, beg);
		out << inch <<"\n";
	}

    return true;
}
/**
 * @brief format_task::add_execution_guard_to_kernel
 * Format text
 * Rewrite kernel function (add guard statements based on global_sizes param.)
 * @param ocl_kernel
 * @param global_sizes
 * @return
 */
inline std::string add_execution_guard_to_kernels(
        const std::string &ocl_kernels,		
        const std::array<size_t, 3> &global_sizes)
{
    if (ocl_kernels.empty())
        return "";

    std::string formated_oclkernel;
    std::stringstream rewriten_oclkernel;

    txt_utils::remove_extra_whitespaces(ocl_kernels, formated_oclkernel);
    txt_utils::remove_comments_block(formated_oclkernel, "/*", "*/");
    txt_utils::remove_comments_block(formated_oclkernel, "//", "\n");
    //txt_utils::remove_newline_tabs_carrieg(formated_oclkernel);

    //std::cout<<formated_oclkernel<<std::endl;

    //// Remove any \r,\t ... chars
    //std::copy_if(ocl_kernels.begin(), ocl_kernels.end(),
    //             std::back_inserter(formated_oclkernel),
    //             [](char c) { return c != '\r' && c != '\t' /* && c != '\n'*/; });

    // find all appearances of: __kernelvoid or kernelvoid and replace with:
    // __kernel void or kernel void
    size_t pos = 0;
    while (pos != std::string::npos)
        pos = pre_process_kernel_func(formated_oclkernel);

    if (!copy_include_headers(rewriten_oclkernel, formated_oclkernel))
        return "";

    while (!formated_oclkernel.empty()) {
        auto kernel_func = cut_kernel_function(formated_oclkernel);
        kernel_func = insert_global_size_guard(kernel_func, global_sizes);
        rewriten_oclkernel << kernel_func << "\n";
    }

    return rewriten_oclkernel.str();
}

inline std::string add_execution_guard_to_kernels(
    const std::string& ocl_kernels    )
{
    if (ocl_kernels.empty())
        return "";

    std::string formated_oclkernel;
    std::stringstream rewriten_oclkernel;
    //Format string
    txt_utils::remove_extra_whitespaces(ocl_kernels, formated_oclkernel);
    txt_utils::remove_comments_block(formated_oclkernel, "/*", "*/");
    txt_utils::remove_comments_block(formated_oclkernel, "//", "\n");
    //txt_utils::remove_newline_tabs_carrieg(formated_oclkernel);
    //std::cout<<formated_oclkernel<<std::endl;
    
    // find all appearances of: __kernelvoid or kernelvoid and replace with:
    // __kernel void or kernel void
    size_t pos = 0;
    while (pos != std::string::npos)
        pos = pre_process_kernel_func(formated_oclkernel);

    if (!copy_include_headers(rewriten_oclkernel, formated_oclkernel))
        return "";

    const std::array<std::string,3> global_size_args = { "_GX_","_GY_","_GZ_" };

    while (!formated_oclkernel.empty()) 
    {
        auto kernel_func = cut_kernel_function(formated_oclkernel);
        kernel_func = insert_global_size_guard_and_args(kernel_func, global_size_args);
        rewriten_oclkernel << kernel_func << "\n";
    }

    return rewriten_oclkernel.str();
}

}
