#include "ggmlc/vmm_pool.h"

namespace ggmlc {

struct VMMBlockManager::Impl {};

VMMBlockManager::VMMBlockManager() = default;
VMMBlockManager::~VMMBlockManager() = default;
VMMBlockManager::VMMBlockManager(VMMBlockManager&&) noexcept = default;
VMMBlockManager& VMMBlockManager::operator=(VMMBlockManager&&) noexcept = default;

bool VMMBlockManager::is_supported_on_device(int) { return false; }
bool VMMBlockManager::init(int) { return false; }
bool VMMBlockManager::is_initialized() const { return false; }
size_t VMMBlockManager::page_size() const { return 0; }

uint64_t VMMBlockManager::reserve_virtual_window(size_t) { return 0; }
void VMMBlockManager::free_virtual_window(uint64_t, size_t) {}

uint64_t VMMBlockManager::alloc_physical_page() { return 0; }
void VMMBlockManager::retain_physical_page(uint64_t) {}
void VMMBlockManager::release_physical_page(uint64_t) {}

bool VMMBlockManager::map_page(uint64_t, uint64_t) { return false; }
bool VMMBlockManager::unmap_page(uint64_t) { return false; }

uint64_t VMMBlockManager::get_prefix_page(const std::string&) { return 0; }
void VMMBlockManager::register_prefix_page(const std::string&, uint64_t) {}
void VMMBlockManager::evict_prefix_page(const std::string&) {}
size_t VMMBlockManager::prefix_cache_size() const { return 0; }

size_t VMMBlockManager::total_reserved_va_bytes() const { return 0; }
size_t VMMBlockManager::total_mapped_physical_bytes() const { return 0; }
size_t VMMBlockManager::total_allocated_pages() const { return 0; }

void VMMBlockManager::reset() {}

} // namespace ggmlc
