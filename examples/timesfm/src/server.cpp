#include "server.h"
#include "web_assets.h"
#include "export.h"
#include "backtest.h"
#include "data_loader.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cctype>

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

namespace timesfm {

static std::vector<float> extract_json_array(const std::string& body, const std::string& key) {
    std::vector<float> values;
    size_t key_pos = body.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return values;

    size_t bracket_start = body.find('[', key_pos);
    if (bracket_start == std::string::npos) return values;

    size_t bracket_end = body.find(']', bracket_start);
    if (bracket_end == std::string::npos) return values;

    std::string arr_str = body.substr(bracket_start + 1, bracket_end - bracket_start - 1);
    std::stringstream ss(arr_str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        size_t s = 0;
        while (s < token.size() && (std::isspace(static_cast<unsigned char>(token[s])) || token[s] == '\r' || token[s] == '\n')) s++;
        size_t e = token.size();
        while (e > s && (std::isspace(static_cast<unsigned char>(token[e - 1])) || token[e - 1] == '\r' || token[e - 1] == '\n')) e--;
        std::string num_str = token.substr(s, e - s);
        if (!num_str.empty()) {
            try {
                values.push_back(std::stof(num_str));
            } catch (...) {}
        }
    }

    return values;
}

static std::vector<std::vector<float>> extract_json_batch(const std::string& body, const std::string& key) {
    std::vector<std::vector<float>> batch;
    size_t key_pos = body.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return batch;

    size_t start = body.find('[', key_pos);
    if (start == std::string::npos) return batch;

    size_t cur = start + 1;
    while (cur < body.size()) {
        size_t inner_start = body.find('[', cur);
        if (inner_start == std::string::npos) break;
        size_t inner_end = body.find(']', inner_start);
        if (inner_end == std::string::npos) break;

        std::string arr_str = body.substr(inner_start + 1, inner_end - inner_start - 1);
        std::stringstream ss(arr_str);
        std::string token;
        std::vector<float> series;
        while (std::getline(ss, token, ',')) {
            size_t s = 0;
            while (s < token.size() && (std::isspace(static_cast<unsigned char>(token[s])) || token[s] == '\r' || token[s] == '\n')) s++;
            size_t e = token.size();
            while (e > s && (std::isspace(static_cast<unsigned char>(token[e - 1])) || token[e - 1] == '\r' || token[e - 1] == '\n')) e--;
            std::string num_str = token.substr(s, e - s);
            if (!num_str.empty()) {
                try {
                    series.push_back(std::stof(num_str));
                } catch (...) {}
            }
        }
        if (!series.empty()) {
            batch.push_back(std::move(series));
        }
        cur = inner_end + 1;
        size_t next_char = cur;
        while (next_char < body.size() && (std::isspace(static_cast<unsigned char>(body[next_char])) || body[next_char] == ',')) next_char++;
        if (next_char < body.size() && body[next_char] == ']') break;
    }
    return batch;
}

static int64_t extract_json_int(const std::string& body, const std::string& key, int64_t default_val) {
    size_t pos = body.find("\"" + key + "\"");
    if (pos == std::string::npos) return default_val;
    size_t colon = body.find(':', pos);
    if (colon == std::string::npos) return default_val;

    size_t val_start = colon + 1;
    while (val_start < body.size() && std::isspace(static_cast<unsigned char>(body[val_start]))) val_start++;

    size_t val_end = val_start;
    while (val_end < body.size() && (std::isdigit(static_cast<unsigned char>(body[val_end])) || body[val_end] == '-')) val_end++;

    try {
        return std::stoll(body.substr(val_start, val_end - val_start));
    } catch (...) {
        return default_val;
    }
}

static bool extract_json_bool(const std::string& body, const std::string& key, bool default_val) {
    size_t pos = body.find("\"" + key + "\"");
    if (pos == std::string::npos) return default_val;
    size_t colon = body.find(':', pos);
    if (colon == std::string::npos) return default_val;

    size_t val_start = colon + 1;
    while (val_start < body.size() && std::isspace(static_cast<unsigned char>(body[val_start]))) val_start++;

    if (body.substr(val_start, 4) == "true") return true;
    if (body.substr(val_start, 5) == "false") return false;
    return default_val;
}

static bool send_all(SOCKET fd, const std::string& data) {
    size_t total_sent = 0;
    while (total_sent < data.size()) {
        int to_send = static_cast<int>(std::min<size_t>(data.size() - total_sent, 65536));
        int sent = send(fd, data.c_str() + total_sent, to_send, 0);
        if (sent <= 0) {
            return false;
        }
        total_sent += sent;
    }
    return true;
}

Server::Server(TimesFMForecaster& forecaster, int port)
    : forecaster_(forecaster), port_(port), running_(false) {}

Server::~Server() {
    stop();
}

void Server::stop() {
    running_ = false;
}

std::string Server::handle_request(const std::string& method, const std::string& path, const std::string& body) {
    if (path == "/favicon.ico") {
        return "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    }

    if (method == "GET" && (path == "/" || path == "/index.html")) {
        std::string html = get_index_html();
        std::string response =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "Content-Length: " + std::to_string(html.size()) + "\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n" + html;
        return response;
    }

    if (method == "GET" && path == "/api/health") {
        std::string json = "{\"status\": \"ok\", \"model\": \"TimesFM 3.0\", \"device\": \"ready\"}\n";
        std::string response =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(json.size()) + "\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n" + json;
        return response;
    }

    if (method == "GET" && path == "/api/presets") {
        std::ostringstream oss;
        oss << "[\n";
        auto presets = DataLoader::get_preset_names();
        for (size_t i = 0; i < presets.size(); ++i) {
            oss << "  \"" << presets[i] << "\"" << (i + 1 < presets.size() ? "," : "") << "\n";
        }
        oss << "]\n";
        std::string json = oss.str();
        std::string response =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(json.size()) + "\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n" + json;
        return response;
    }

    if (method == "POST" && path == "/api/forecast") {
        std::vector<float> context = extract_json_array(body, "context");
        if (context.empty()) {
            // Check if preset or text was specified
            size_t preset_pos = body.find("\"preset\"");
            if (preset_pos != std::string::npos) {
                size_t quote_start = body.find('"', body.find(':', preset_pos));
                if (quote_start != std::string::npos) {
                    size_t quote_end = body.find('"', quote_start + 1);
                    if (quote_end != std::string::npos) {
                        std::string p_name = body.substr(quote_start + 1, quote_end - quote_start - 1);
                        TimeSeriesData ts;
                        if (DataLoader::load_preset(p_name, ts, 192)) {
                            context = ts.values;
                        }
                    }
                }
            }
        }

        if (context.empty()) {
            std::string err_json = "{\"error\": \"Missing or empty 'context' float array in JSON body.\"}\n";
            return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: " +
                   std::to_string(err_json.size()) + "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n" + err_json;
        }

        ForecastConfig cfg;
        cfg.horizon = extract_json_int(body, "horizon", 128);
        cfg.normalize = extract_json_bool(body, "normalize", true);
        cfg.detrend = extract_json_bool(body, "detrend", true);
        cfg.sort_quantiles = extract_json_bool(body, "sort_quantiles", true);
        cfg.make_positive = extract_json_bool(body, "make_positive", false) || extract_json_bool(body, "non_negative", false);
        cfg.use_symmetric_averaging = extract_json_bool(body, "use_symmetric_averaging", false) || extract_json_bool(body, "sym_avg", false);

        std::cout << "[TimesFM Server] Running forecast for " << context.size() << " context points (horizon=" << cfg.horizon << ")..." << std::endl;
        ForecastResult result = forecaster_.forecast(context, cfg);
        std::string res_json = Exporter::to_json_string(result, context);

        std::string response =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(res_json.size()) + "\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n" + res_json;
        return response;
    }

    if (method == "POST" && path == "/api/forecast_batch") {
        std::vector<std::vector<float>> batch = extract_json_batch(body, "batch");
        if (batch.empty()) {
            std::string err_json = "{\"error\": \"Missing or empty 'batch' 2D float array in JSON body.\"}\n";
            return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: " +
                   std::to_string(err_json.size()) + "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n" + err_json;
        }

        ForecastConfig cfg;
        cfg.horizon = extract_json_int(body, "horizon", 128);
        cfg.normalize = extract_json_bool(body, "normalize", true);
        cfg.detrend = extract_json_bool(body, "detrend", true);
        cfg.sort_quantiles = extract_json_bool(body, "sort_quantiles", true);
        cfg.make_positive = extract_json_bool(body, "make_positive", false);
        cfg.use_symmetric_averaging = extract_json_bool(body, "use_symmetric_averaging", false);

        std::cout << "[TimesFM Server] Running batch forecast for " << batch.size() << " series (horizon=" << cfg.horizon << ")..." << std::endl;
        std::vector<ForecastResult> results = forecaster_.forecast_batch(batch, cfg);
        std::ostringstream oss;
        oss << "[\n";
        for (size_t i = 0; i < results.size(); ++i) {
            oss << Exporter::to_json_string(results[i], batch[i]);
            if (i + 1 < results.size()) oss << ",\n";
        }
        oss << "\n]";
        std::string res_json = oss.str();

        std::string response =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(res_json.size()) + "\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n" + res_json;
        return response;
    }

    if (method == "POST" && path == "/api/backtest") {
        std::vector<float> series = extract_json_array(body, "series");
        if (series.empty()) {
            // Check preset
            size_t preset_pos = body.find("\"preset\"");
            if (preset_pos != std::string::npos) {
                size_t quote_start = body.find('"', body.find(':', preset_pos));
                if (quote_start != std::string::npos) {
                    size_t quote_end = body.find('"', quote_start + 1);
                    if (quote_end != std::string::npos) {
                        std::string p_name = body.substr(quote_start + 1, quote_end - quote_start - 1);
                        TimeSeriesData ts;
                        if (DataLoader::load_preset(p_name, ts, 192)) {
                            series = ts.values;
                        }
                    }
                }
            }
        }

        if (series.empty()) {
            std::string err_json = "{\"error\": \"Missing or empty 'series' float array in JSON body.\"}\n";
            return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: " +
                   std::to_string(err_json.size()) + "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n" + err_json;
        }

        BacktestConfig cfg;
        cfg.context_len = extract_json_int(body, "context_len", 128);
        cfg.horizon = extract_json_int(body, "horizon", 64);
        cfg.stride = extract_json_int(body, "stride", 64);
        cfg.max_windows = extract_json_int(body, "max_windows", 32);
        cfg.normalize = extract_json_bool(body, "normalize", true);
        cfg.detrend = extract_json_bool(body, "detrend", true);
        cfg.sort_quantiles = extract_json_bool(body, "sort_quantiles", true);
        cfg.make_positive = extract_json_bool(body, "make_positive", false);

        std::cout << "[TimesFM Server] Running rolling backtest for series of length " << series.size() << "..." << std::endl;
        BacktestResult bt_res = BacktestEngine::run_backtest(forecaster_, series, cfg);
        std::string res_json = BacktestEngine::to_json_string(bt_res, series);

        std::string response =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(res_json.size()) + "\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n" + res_json;
        return response;
    }

    if (method == "OPTIONS") {
        return "HTTP/1.1 204 No Content\r\n"
               "Access-Control-Allow-Origin: *\r\n"
               "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
               "Access-Control-Allow-Headers: Content-Type\r\n"
               "Content-Length: 0\r\n"
               "Connection: close\r\n\r\n";
    }

    std::string not_found = "{\"error\": \"Endpoint not found\"}\n";
    return "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: " +
           std::to_string(not_found.size()) + "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n" + not_found;
}

bool Server::start() {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::cerr << "WSAStartup failed." << std::endl;
        return false;
    }
#endif

    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) {
        std::cerr << "Socket creation failed." << std::endl;
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

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(static_cast<uint16_t>(port_));

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == SOCKET_ERROR) {
        std::cerr << "Bind failed on port " << port_ << std::endl;
        closesocket(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    if (listen(server_fd, 10) == SOCKET_ERROR) {
        std::cerr << "Listen failed on port " << port_ << std::endl;
        closesocket(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }

    running_ = true;
    std::cout << "\n======================================================================" << std::endl;
    std::cout << " TimesFM 3.0 Web Dashboard & REST API Server Started" << std::endl;
    std::cout << " -> Web UI URL : http://localhost:" << port_ << "/" << std::endl;
    std::cout << " -> REST API   : http://localhost:" << port_ << "/api/forecast" << std::endl;
    std::cout << " -> Health     : http://localhost:" << port_ << "/api/health" << std::endl;
    std::cout << "======================================================================\n" << std::endl;

    while (running_) {
        sockaddr_in client_addr;
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
                    std::transform(req_lower.begin(), req_lower.end(), req_lower.begin(), [](unsigned char c) { return std::tolower(c); });
                    size_t cl_pos = req_lower.find("content-length:");
                    if (cl_pos != std::string::npos) {
                        size_t val_start = cl_pos + 15;
                        while (val_start < req_lower.size() && (req_lower[val_start] == ' ' || req_lower[val_start] == '\t')) val_start++;
                        size_t val_end = val_start;
                        while (val_end < req_lower.size() && std::isdigit(static_cast<unsigned char>(req_lower[val_end]))) val_end++;
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
                size_t current_body_len = req.size() - body_start;
                if (current_body_len >= content_length) {
                    break;
                }
            }
        }

        if (!req.empty()) {
            std::stringstream ss(req);
            std::string method, path, version;
            ss >> method >> path >> version;

            std::string body;
            size_t body_pos = req.find("\r\n\r\n");
            if (body_pos != std::string::npos) {
                body = req.substr(body_pos + 4);
            }

            std::cout << "[TimesFM Server] " << method << " " << path;
            if (!body.empty()) {
                std::cout << " (" << body.size() << " bytes)";
            }
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

} // namespace timesfm

