#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

namespace ggmlc {

class VMMBlockManager {
public:
    VMMBlockManager();
    ~VMMBlockManager();

    VMMBlockManager(const VMMBlockManager&) = delete;
    VMMBlockManager& operator=(const VMMBlockManager&) = delete;
    VMMBlockManager(VMMBlockManager&&) noexcept;
    VMMBlockManager& operator=(VMMBlockManager&&) noexcept;

    // Check device support and initialize VMM driver interface
    static bool is_supported_on_device(int device_id);
    bool init(int device_id);
    bool is_initialized() const;

    // Allocation granularity (e.g. 2 MB)
    size_t page_size() const;

    // Virtual Address Window Management (64-bit GPU virtual address space)
    uint64_t reserve_virtual_window(size_t window_bytes);
    void free_virtual_window(uint64_t va_ptr, size_t window_bytes);

    // Physical Page Allocation & Lifecycle Management
    uint64_t alloc_physical_page();
    void retain_physical_page(uint64_t page_handle);
    void release_physical_page(uint64_t page_handle);

    // Page Mapping into Virtual Address Slots
    bool map_page(uint64_t va_offset, uint64_t page_handle);
    bool unmap_page(uint64_t va_offset);

    // Prefix Caching (Zero-Copy 1-to-N Physical Page Sharing)
    uint64_t get_prefix_page(const std::string& prefix_hash);
    void register_prefix_page(const std::string& prefix_hash, uint64_t page_handle);
    void evict_prefix_page(const std::string& prefix_hash);
    size_t prefix_cache_size() const;

    // Warm Pool & Upfront Allocation Management
    void configure_pool(size_t max_warm_pages, size_t prealloc_pages = 0);
    size_t warm_pool_pages() const;
    size_t free_pool_pages() const;
    size_t max_warm_pages() const;
    void drain_warm_pool();

    // Telemetry & Accounting
    size_t total_reserved_va_bytes() const;
    size_t total_mapped_physical_bytes() const;
    size_t total_allocated_pages() const;

    // Reset all mappings and allocations
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ggmlc
