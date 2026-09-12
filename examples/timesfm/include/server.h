#pragma once

#include <string>
#include <memory>
#include <cstdint>
#include "forecaster.h"

namespace timesfm {

class Server {
public:
    explicit Server(TimesFMForecaster& forecaster, int port = 8080);
    ~Server();

    // Starts the blocking HTTP server loop
    bool start();

    // Stops the server
    void stop();

private:
    TimesFMForecaster& forecaster_;
    int port_ = 8080;
    bool running_ = false;

    std::string handle_request(const std::string& method, const std::string& path, const std::string& body);
};

} // namespace timesfm
