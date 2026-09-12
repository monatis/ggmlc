#pragma once

#include <string>
#include <vector>
#include "export.h"

namespace timesfm {

class Visualizer {
public:
    // Generates a standalone, beautiful SVG chart with dark aesthetic and shaded quantile bands.
    static bool generate_svg(
        const std::string& filepath,
        const std::vector<float>& history,
        const ForecastResult& forecast,
        const std::string& title = "Google TimesFM 3.0 Forecast",
        int width = 1000,
        int height = 500
    );

    // Generates a standalone, interactive HTML chart with Chart.js.
    static bool generate_html_chart(
        const std::string& filepath,
        const std::vector<float>& history,
        const ForecastResult& forecast,
        const std::string& title = "Google TimesFM 3.0 Interactive Forecast"
    );
};

} // namespace timesfm
