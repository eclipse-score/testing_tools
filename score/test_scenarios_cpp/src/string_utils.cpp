// *******************************************************************************
// Copyright (c) 2025 Contributors to the Eclipse Foundation
//
// See the NOTICE file(s) distributed with this work for additional
// information regarding copyright ownership.
//
// This program and the accompanying materials are made available under the
// terms of the Apache License Version 2.0 which is available at
// https://www.apache.org/licenses/LICENSE-2.0
//
// SPDX-License-Identifier: Apache-2.0
// *******************************************************************************
#include "string_utils.hpp"

#include <algorithm>
#include <sstream>

std::vector<std::string> string_utils::split(const std::string& str, const std::string& delimiter) {
    const auto delimiter_size{delimiter.length()};
    std::vector<std::string> parts;
    size_t pos_begin{0};
    size_t pos_end{0};

    while (true) {
        // Find next delimiter occurrence.
        pos_end = str.find(delimiter, pos_begin);

        // Make next substring.
        std::string part{str.substr(pos_begin, pos_end - pos_begin)};
        parts.push_back(part);

        // End execution if end of string reached.
        if (pos_end == std::string::npos) {
            break;
        }

        // Recalculate string start pos - pos_end is the position of delimiter.
        pos_begin = pos_end + delimiter_size;
    }

    return parts;
}

std::string string_utils::join(const std::vector<std::string>& parts,
                               const std::string& delimiter) {
    std::stringstream ss;
    for (size_t i = 0; i < parts.size(); ++i) {
        ss << parts[i];
        if (i < parts.size() - 1) {
            ss << ".";
        }
    }
    return ss.str();
}

std::string string_utils::trim(const std::string& str) {
    auto fn = [](int c) { return !std::isspace(c); };
    auto left_pos{std::find_if(str.begin(), str.end(), fn)};
    auto right_pos{std::find_if(str.rbegin(), str.rend(), fn).base()};
    return std::string{left_pos, right_pos};
}
