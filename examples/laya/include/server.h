#pragma once

#include "engine.h"
#include "questions.h"

namespace laya {

class DecisionRouter;

class Server {
public:
    Server(DecisionEngine& engine, int port = 8080);
    Server(DecisionRouter& router, int port = 8080);
    ~Server();
    bool start();
    void stop();

private:
    DecisionEngine* engine_ = nullptr;
    DecisionRouter* router_ = nullptr;
    int port_ = 8080;
    bool running_ = false;
    std::string handle_request(const std::string& method, const std::string& path, const std::string& body);
    std::string device() const;
    DecideResult decide(const JsonValue& state, const std::vector<Question>& qs);
};

}  // namespace laya
