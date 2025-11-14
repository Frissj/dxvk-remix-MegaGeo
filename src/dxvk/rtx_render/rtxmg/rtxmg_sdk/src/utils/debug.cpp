/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#include <cstring>
#include <fstream>
#include <sstream>
#include <filesystem>

#include <iostream>

int GetUniqueFileIndex(const char* baseName, const char* extension)
{
    // Avoid overwriting an existing screenshot : scan the default output directory
    // for existing files with pattern 'screenshot_xxxx.bmp' to find the highest index.
    int index = -1;
    namespace fs = std::filesystem;
    for (auto it : fs::directory_iterator(fs::current_path()))
    {
        if (it.path().extension() != extension)
            continue;
        std::string filename = it.path().filename().generic_string();
        if (std::strstr(filename.c_str(), baseName) != filename.c_str())
            continue;
        int existingIndex = std::atoi(filename.c_str() + strlen(baseName));
        index = std::max(index, existingIndex);
    }
    return index + 1;
}
