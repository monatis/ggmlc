#include "server.h"
#include "web_assets.h"
#include "presets.h"
#include "questions.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef int socklen_t;
#define SHUT_SEND SD_SEND
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#define closesocket close
typedef int SOCKET;
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define SHUT_SEND SHUT_WR
#endif

namespace laya {

static bool send_all(SOCKET fd, const std::string& data) {
    size_t total = 0;
    while (total < data.size()) {
        int n = static_cast<int>(std::min<size_t>(data.size() - total, 65536));
        int sent = send(fd, data.c_str() + total, n, 0);
        if (sent <= 0) return false;
        total += static_cast<size_t>(sent);
    }
    return true;
}

static std::string http_json(int code, const std::string& body, const char* status = "OK") {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << code << " " << status << "\r\n"
        << "Content-Type: application/json; charset=utf-8\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    return oss.str();
}

static std::string http_html(const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: text/html; charset=utf-8\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    return oss.str();
}

Server::Server(DecisionEngine& engine, int port) : engine_(engine), port_(port), running_(false) {}
Server::~Server() { stop(); }
void Server::stop() { running_ = false; }

std::string Server::handle_request(const std::string& method, const std::string& path, const std::string& body) {
    std::string p = path;
    auto qpos = p.find('?');
    if (qpos != std::string::npos) p = p.substr(0, qpos);

    if (p == "/favicon.ico") {
        return "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    }
    if (method == "OPTIONS") {
        return "HTTP/1.1 204 No Content\r\n"
               "Access-Control-Allow-Origin: *\r\n"
               "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
               "Access-Control-Allow-Headers: Content-Type\r\n"
               "Content-Length: 0\r\nConnection: close\r\n\r\n";
    }
    if (method == "GET" && (p == "/" || p == "/index.html")) {
        return http_html(get_index_html());
    }
    if (method == "GET" && p == "/api/health") {
        JsonValue o = JsonValue::object();
        o.set("status", JsonValue::string("ok"));
        o.set("model", JsonValue::string("laya"));
        o.set("device", JsonValue::string(engine_.device()));
        return http_json(200, json_dumps(o) + "\n");
    }
    if (method == "GET" && p == "/api/presets") {
        JsonValue arr = JsonValue::array();
        for (const auto& pr : all_presets()) {
            JsonValue o = JsonValue::object();
            o.set("name", JsonValue::string(pr.name));
            o.set("title", JsonValue::string(pr.title));
            o.set("blurb", JsonValue::string(pr.blurb));
            o.set("state_key", JsonValue::string(pr.state_key));
            o.set("state", pr.default_state);
            o.set("questions", questions_to_json(pr.questions));
            arr.arr.push_back(std::move(o));
        }
        return http_json(200, json_dumps_pretty(arr) + "\n");
    }
    if (method == "POST" && (p == "/api/decide" || p == "/v1/decide" || p == "/system_one")) {
        JsonValue req;
        try {
            req = JsonParser::parse_string(body.empty() ? "{}" : body);
        } catch (const std::exception& e) {
            JsonValue err = JsonValue::object();
            err.set("error", JsonValue::string(e.what()));
            return http_json(400, json_dumps(err) + "\n", "Bad Request");
        }
        JsonValue state = JsonValue::object();
        if (const JsonValue* s = req.get("state")) state = *s;
        std::vector<Question> qs;
        if (const JsonValue* q = req.get("questions")) qs = questions_from_json(*q);
        if (const JsonValue* pn = req.get("preset")) {
            if (pn->is_string()) {
                const Preset* pr = find_preset(pn->s);
                if (pr) {
                    if (qs.empty()) qs = pr->questions;
                    if (state.is_object() && state.obj.empty()) state = pr->default_state;
                }
            }
        }
        if (const JsonValue* t = req.get("text")) {
            if (t->is_string()) {
                const Preset* pr = nullptr;
                if (const JsonValue* pn = req.get("preset")) {
                    if (pn->is_string()) pr = find_preset(pn->s);
                }
                state = apply_text_to_state(pr, state, t->s);
            }
        }
        if (qs.empty()) {
            JsonValue err = JsonValue::object();
            err.set("error", JsonValue::string("missing questions"));
            return http_json(400, json_dumps(err) + "\n", "Bad Request");
        }
        DecideResult r = engine_.decide(state, qs);
        return http_json(200, format_answer_json(r, true) + "\n");
    }

    JsonValue err = JsonValue::object();
    err.set("error", JsonValue::string("Endpoint not found"));
    return http_json(404, json_dumps(err) + "\n", "Not Found");
}

bool Server::start() {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::cerr << "WSAStartup failed.\n";
        return false;
    }
#endif
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) {
        std::cerr << "Socket creation failed.\n";
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }
    int opt = 1;
#ifdef _WIN32
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(static_cast<uint16_t>(port_));
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == SOCKET_ERROR) {
        std::cerr << "Bind failed on port " << port_ << "\n";
        closesocket(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }
    if (listen(server_fd, 10) == SOCKET_ERROR) {
        std::cerr << "Listen failed on port " << port_ << "\n";
        closesocket(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }
    running_ = true;
    std::cout << "\n======================================================================\n"
              << " Laya System 1 Decision Studio\n"
              << " -> Web UI  : http://localhost:" << port_ << "/\n"
              << " -> Decide  : POST http://localhost:" << port_ << "/api/decide\n"
              << " -> Health  : http://localhost:" << port_ << "/api/health\n"
              << "======================================================================\n\n" << std::flush;

    while (running_) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        SOCKET client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd == INVALID_SOCKET) {
            if (!running_) break;
            continue;
        }
#ifdef _WIN32
        DWORD timeout = 8000;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
#else
        struct timeval tv;
        tv.tv_sec = 8;
        tv.tv_usec = 0;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
#endif
        std::string req;
        char buffer[16384];
        size_t content_length = 0;
        bool headers_complete = false;
        while (running_) {
            int bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
            if (bytes_read <= 0) break;
            req.append(buffer, bytes_read);
            if (!headers_complete) {
                size_t header_end = req.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                    headers_complete = true;
                    std::string req_headers = req.substr(0, header_end);
                    std::string req_lower = req_headers;
                    std::transform(req_lower.begin(), req_lower.end(), req_lower.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    size_t cl_pos = req_lower.find("content-length:");
                    if (cl_pos != std::string::npos) {
                        size_t val_start = cl_pos + 15;
                        while (val_start < req_lower.size() &&
                               (req_lower[val_start] == ' ' || req_lower[val_start] == '\t'))
                            val_start++;
                        size_t val_end = val_start;
                        while (val_end < req_lower.size() &&
                               std::isdigit(static_cast<unsigned char>(req_lower[val_end])))
                            val_end++;
                        try {
                            content_length = std::stoull(req_lower.substr(val_start, val_end - val_start));
                        } catch (...) {
                            content_length = 0;
                        }
                    }
                }
            }
            if (headers_complete) {
                size_t body_start = req.find("\r\n\r\n") + 4;
                if (req.size() - body_start >= content_length) break;
            }
        }
        if (!req.empty()) {
            std::stringstream ss(req);
            std::string method, path, version;
            ss >> method >> path >> version;
            std::string body;
            size_t body_pos = req.find("\r\n\r\n");
            if (body_pos != std::string::npos) body = req.substr(body_pos + 4);
            std::cout << "[laya] " << method << " " << path;
            if (!body.empty()) std::cout << " (" << body.size() << " bytes)";
            std::cout << std::endl;
            std::string response = handle_request(method, path, body);
            send_all(client_fd, response);
            shutdown(client_fd, SHUT_SEND);
        }
        closesocket(client_fd);
    }
    closesocket(server_fd);
#ifdef _WIN32
    WSACleanup();
#endif
    return true;
}

}  // namespace laya
