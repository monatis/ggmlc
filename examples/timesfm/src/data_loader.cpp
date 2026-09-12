#include "data_loader.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cctype>

namespace timesfm {

static std::string trim(const std::string& str) {
    size_t start = 0;
    while (start < str.size() && (std::isspace(static_cast<unsigned char>(str[start])) || str[start] == '\r' || str[start] == '"')) {
        start++;
    }
    size_t end = str.size();
    while (end > start && (std::isspace(static_cast<unsigned char>(str[end - 1])) || str[end - 1] == '\r' || str[end - 1] == '"')) {
        end--;
    }
    return str.substr(start, end - start);
}

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool inside_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            inside_quotes = !inside_quotes;
        } else if (c == ',' && !inside_quotes) {
            fields.push_back(trim(current));
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    fields.push_back(trim(current));
    return fields;
}

static bool is_numeric(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

bool DataLoader::load_csv(
    const std::string& filepath,
    const std::string& target_col,
    TimeSeriesData& out_data,
    std::string& err_msg
) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        err_msg = "Could not open CSV file: " + filepath;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return parse_csv_content(buffer.str(), target_col, out_data, err_msg);
}

bool DataLoader::parse_csv_content(
    const std::string& content,
    const std::string& target_col,
    TimeSeriesData& out_data,
    std::string& err_msg
) {
    out_data.timestamps.clear();
    out_data.values.clear();
    out_data.original_count = 0;
    out_data.missing_count = 0;

    std::stringstream ss(content);
    std::string line;

    if (!std::getline(ss, line)) {
        err_msg = "CSV content is empty.";
        return false;
    }

    std::vector<std::string> headers = split_csv_line(line);
    if (headers.empty()) {
        err_msg = "CSV contains no headers or columns.";
        return false;
    }

    int date_col_idx = -1;
    int val_col_idx = -1;

    // Detect timestamp/date column
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string h_lower = headers[i];
        std::transform(h_lower.begin(), h_lower.end(), h_lower.begin(), ::tolower);
        if (h_lower == "date" || h_lower == "timestamp" || h_lower == "time" || h_lower == "datetime" || h_lower == "ds") {
            date_col_idx = static_cast<int>(i);
            break;
        }
    }

    // Detect target value column
    if (!target_col.empty()) {
        for (size_t i = 0; i < headers.size(); ++i) {
            if (headers[i] == target_col) {
                val_col_idx = static_cast<int>(i);
                break;
            }
        }
        if (val_col_idx == -1) {
            err_msg = "Target column '" + target_col + "' not found in CSV headers.";
            return false;
        }
    } else {
        // Find first non-date numeric column or second column if date is column 0
        for (size_t i = 0; i < headers.size(); ++i) {
            if (static_cast<int>(i) != date_col_idx) {
                val_col_idx = static_cast<int>(i);
                break;
            }
        }
    }

    if (val_col_idx == -1) {
        err_msg = "Could not identify a value column in CSV.";
        return false;
    }

    out_data.column_name = headers[val_col_idx];

    int line_idx = 1;
    while (std::getline(ss, line)) {
        line = trim(line);
        if (line.empty()) continue;

        std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() <= static_cast<size_t>(val_col_idx)) {
            continue;
        }

        std::string ts = (date_col_idx >= 0 && static_cast<size_t>(date_col_idx) < fields.size())
                             ? fields[date_col_idx]
                             : std::to_string(line_idx);

        std::string val_str = fields[val_col_idx];
        float val = 0.0f;
        if (val_str.empty() || val_str == "NA" || val_str == "null" || val_str == "NULL" || val_str == "NaN") {
            val = std::numeric_limits<float>::quiet_NaN();
            out_data.missing_count++;
        } else {
            try {
                val = std::stof(val_str);
            } catch (...) {
                val = std::numeric_limits<float>::quiet_NaN();
                out_data.missing_count++;
            }
        }

        out_data.timestamps.push_back(ts);
        out_data.values.push_back(val);
        out_data.original_count++;
        line_idx++;
    }

    if (out_data.values.empty()) {
        err_msg = "No data rows found in CSV.";
        return false;
    }

    return true;
}

bool DataLoader::parse_series_text(
    const std::string& text,
    TimeSeriesData& out_data,
    std::string& err_msg
) {
    out_data.timestamps.clear();
    out_data.values.clear();
    out_data.column_name = "series_text";
    out_data.original_count = 0;
    out_data.missing_count = 0;

    if (text.empty()) {
        err_msg = "Empty text input provided.";
        return false;
    }

    std::string token;
    auto flush_token = [&](std::string t) {
        t = trim(t);
        if (t.empty()) return;
        try {
            float val = std::stof(t);
            if (std::isfinite(val)) {
                out_data.values.push_back(val);
                out_data.timestamps.push_back(std::to_string(out_data.values.size()));
                out_data.original_count++;
            }
        } catch (...) {
            // ignore non-numeric tokens
        }
    };

    for (char c : text) {
        if (c == ',' || c == ';' || c == '|' || c == '\n' || c == '\r' || c == '\t' || c == ' ') {
            if (!token.empty()) {
                flush_token(token);
                token.clear();
            }
        } else {
            token.push_back(c);
        }
    }
    if (!token.empty()) {
        flush_token(token);
    }

    if (out_data.values.size() < 8) {
        err_msg = "Need at least 8 numeric values for time series forecasting (found " + std::to_string(out_data.values.size()) + ").";
        return false;
    }

    return true;
}

std::vector<std::string> DataLoader::get_preset_names() {
    return {
        "linear_trend",
        "seasonal_sine",
        "trend_seasonal",
        "random_walk",
        "weekly_retail",
        "spiky_demand",
        "airline_passengers",
        "sunspots"
    };
}

bool DataLoader::load_preset(
    const std::string& name,
    TimeSeriesData& out_data,
    int length
) {
    out_data.timestamps.clear();
    out_data.values.clear();
    out_data.column_name = name;
    out_data.original_count = 0;
    out_data.missing_count = 0;

    std::string p_name = name;
    std::transform(p_name.begin(), p_name.end(), p_name.begin(), ::tolower);

    if (p_name == "airline_passengers") {
        static const float air_passengers[] = {
            112.0f, 118.0f, 132.0f, 129.0f, 121.0f, 135.0f, 148.0f, 148.0f, 136.0f, 119.0f, 104.0f, 118.0f,
            115.0f, 126.0f, 141.0f, 135.0f, 125.0f, 149.0f, 170.0f, 170.0f, 158.0f, 133.0f, 114.0f, 140.0f,
            145.0f, 150.0f, 178.0f, 163.0f, 172.0f, 178.0f, 199.0f, 199.0f, 184.0f, 162.0f, 146.0f, 166.0f,
            171.0f, 180.0f, 193.0f, 181.0f, 183.0f, 218.0f, 230.0f, 242.0f, 209.0f, 191.0f, 172.0f, 194.0f,
            196.0f, 196.0f, 236.0f, 235.0f, 229.0f, 243.0f, 264.0f, 272.0f, 237.0f, 211.0f, 180.0f, 201.0f,
            204.0f, 188.0f, 235.0f, 227.0f, 234.0f, 264.0f, 302.0f, 293.0f, 259.0f, 229.0f, 203.0f, 229.0f,
            242.0f, 233.0f, 267.0f, 269.0f, 270.0f, 315.0f, 364.0f, 347.0f, 312.0f, 274.0f, 237.0f, 278.0f,
            284.0f, 277.0f, 317.0f, 313.0f, 318.0f, 374.0f, 413.0f, 405.0f, 355.0f, 306.0f, 271.0f, 306.0f,
            315.0f, 301.0f, 356.0f, 348.0f, 355.0f, 422.0f, 465.0f, 467.0f, 404.0f, 347.0f, 305.0f, 336.0f,
            340.0f, 318.0f, 362.0f, 348.0f, 363.0f, 435.0f, 491.0f, 505.0f, 404.0f, 359.0f, 310.0f, 337.0f,
            360.0f, 342.0f, 406.0f, 396.0f, 420.0f, 472.0f, 548.0f, 559.0f, 463.0f, 407.0f, 362.0f, 405.0f,
            417.0f, 391.0f, 419.0f, 461.0f, 472.0f, 535.0f, 622.0f, 606.0f, 508.0f, 461.0f, 390.0f, 432.0f
        };
        int n = sizeof(air_passengers) / sizeof(air_passengers[0]);
        for (int i = 0; i < n; ++i) {
            int year = 1949 + i / 12;
            int month = (i % 12) + 1;
            char buf[32];
            snprintf(buf, sizeof(buf), "%04d-%02d", year, month);
            out_data.timestamps.push_back(buf);
            out_data.values.push_back(air_passengers[i]);
        }
        out_data.original_count = n;
        return true;
    }

    if (p_name == "sunspots") {
        // Solar cycle simulation (11-year / 132-month period)
        int n = (length > 0) ? length : 180;
        for (int i = 0; i < n; ++i) {
            float cycle = std::sin(2.0f * 3.14159265f * i / 132.0f);
            float base = (cycle > 0.0f) ? (cycle * cycle * 120.0f) : (cycle * 20.0f);
            float noise = std::sin(i * 1.7f) * 8.0f + std::cos(i * 0.3f) * 5.0f;
            float val = std::max(0.0f, base + noise + 10.0f);
            out_data.timestamps.push_back(std::to_string(i + 1));
            out_data.values.push_back(val);
        }
        out_data.original_count = n;
        return true;
    }

    int n = (length > 0) ? length : 192;
    uint32_t seed = 777;
    auto lcg_rand = [&]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>(seed) / static_cast<float>(0xFFFFFFFFu);
    };
    auto normal_approx = [&]() -> float {
        float sum = 0.0f;
        for (int k = 0; k < 6; ++k) sum += lcg_rand();
        return (sum - 3.0f) * 1.41421356f;
    };

    if (p_name == "linear_trend") {
        for (int t = 0; t < n; ++t) {
            float val = 20.0f + 0.18f * t + normal_approx() * 0.6f;
            out_data.timestamps.push_back(std::to_string(t + 1));
            out_data.values.push_back(val);
        }
    } else if (p_name == "seasonal_sine") {
        for (int t = 0; t < n; ++t) {
            float val = 12.0f + 6.5f * std::sin(2.0f * 3.14159265f * t / 24.0f) + normal_approx() * 0.35f;
            out_data.timestamps.push_back(std::to_string(t + 1));
            out_data.values.push_back(val);
        }
    } else if (p_name == "trend_seasonal") {
        for (int t = 0; t < n; ++t) {
            float val = 40.0f + 0.12f * t
                      + 8.0f * std::sin(2.0f * 3.14159265f * t / 12.0f)
                      + 3.0f * std::sin(2.0f * 3.14159265f * t / 4.0f)
                      + normal_approx() * 1.1f;
            out_data.timestamps.push_back(std::to_string(t + 1));
            out_data.values.push_back(val);
        }
    } else if (p_name == "random_walk") {
        float curr = 50.0f;
        for (int t = 0; t < n; ++t) {
            curr += 0.05f + normal_approx() * 1.1f;
            out_data.timestamps.push_back(std::to_string(t + 1));
            out_data.values.push_back(curr);
        }
    } else if (p_name == "weekly_retail") {
        static const float weekly_mult[7] = {0.70f, 0.75f, 0.80f, 0.85f, 1.15f, 1.45f, 1.30f};
        for (int t = 0; t < n; ++t) {
            float val = (30.0f + 0.08f * t) * weekly_mult[t % 7] + normal_approx() * 1.4f;
            out_data.timestamps.push_back(std::to_string(t + 1));
            out_data.values.push_back(std::max(0.0f, val));
        }
    } else if (p_name == "spiky_demand") {
        for (int t = 0; t < n; ++t) {
            float val = 4.0f + 0.5f * std::sin(2.0f * 3.14159265f * t / 16.0f);
            if (lcg_rand() < 0.08f) {
                val += 8.0f + lcg_rand() * 10.0f;
            }
            val += normal_approx() * 0.25f;
            out_data.timestamps.push_back(std::to_string(t + 1));
            out_data.values.push_back(std::max(0.0f, val));
        }
    } else {
        // Fallback default: trend_seasonal
        for (int t = 0; t < n; ++t) {
            float val = 40.0f + 0.12f * t + 8.0f * std::sin(2.0f * 3.14159265f * t / 12.0f) + normal_approx() * 1.0f;
            out_data.timestamps.push_back(std::to_string(t + 1));
            out_data.values.push_back(val);
        }
    }

    out_data.original_count = out_data.values.size();
    return true;
}

} // namespace timesfm
