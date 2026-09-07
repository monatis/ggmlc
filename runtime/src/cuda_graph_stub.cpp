#include "ggmlc/cuda_graph.h"

namespace ggmlc {

struct CUDAGraphManager::Impl {};

CUDAGraphManager::CUDAGraphManager() = default;
CUDAGraphManager::~CUDAGraphManager() = default;
CUDAGraphManager::CUDAGraphManager(CUDAGraphManager&&) noexcept = default;
CUDAGraphManager& CUDAGraphManager::operator=(CUDAGraphManager&&) noexcept = default;

bool CUDAGraphManager::is_supported_on_device(int) { return false; }
int CUDAGraphManager::get_device_cc(int) { return 0; }
void CUDAGraphManager::init(ggml_backend_t) {}
bool CUDAGraphManager::is_initialized() const { return false; }
bool CUDAGraphManager::is_captured() const { return false; }
bool CUDAGraphManager::begin_capture() { return false; }
bool CUDAGraphManager::end_capture_and_instantiate() { return false; }
bool CUDAGraphManager::begin_capture_bucket(int) { return false; }
bool CUDAGraphManager::end_capture_and_instantiate_bucket(int) { return false; }
bool CUDAGraphManager::launch_bucket(int) { return false; }
bool CUDAGraphManager::is_bucket_captured(int) const { return false; }
bool CUDAGraphManager::update_executable(struct ggml_cgraph*, ggml_backend_t) { return false; }
bool CUDAGraphManager::launch() { return false; }
void CUDAGraphManager::reset() {}

} // namespace ggmlc
