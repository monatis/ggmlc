#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace timesfm {

struct TimeSeriesData {
    std::vector<std::string> timestamps;
    std::vector<float> values;
    std::string column_name;
    int64_t original_count = 0;
    int64_t missing_count = 0;
};

class DataLoader {
public:
    // Loads time series from CSV file.
    // If target_col is empty, selects the first numeric column after the timestamp/index.
    static bool load_csv(
        const std::string& filepath,
        const std::string& target_col,
        TimeSeriesData& out_data,
        std::string& err_msg
    );

    // Parses raw CSV text buffer directly.
    static bool parse_csv_content(
        const std::string& content,
        const std::string& target_col,
        TimeSeriesData& out_data,
        std::string& err_msg
    );

    // Parses freeform pasted text numbers separated by commas, spaces, tabs, semicolons, or newlines.
    static bool parse_series_text(
        const std::string& text,
        TimeSeriesData& out_data,
        std::string& err_msg
    );

    // Returns the list of built-in synthetic and classic benchmark preset names.
    static std::vector<std::string> get_preset_names();

    // Loads a built-in synthetic or classic benchmark preset into out_data.
    static bool load_preset(
        const std::string& name,
        TimeSeriesData& out_data,
        int length = 192
    );
};

} // namespace timesfm
