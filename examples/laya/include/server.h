#pragma once

#include "engine.h"

namespace laya {

class Server {
public:
    Server(DecisionEngine& engine, int port = 8080);
    ~Server();
    bool start();
    void stop();

private:
    DecisionEngine& engine_;
    int port_ = 8080;
    bool running_ = false;
    std::string handle_request(const std::string& method, const std::string& path, const std::string& body);
};

}  // namespace laya
