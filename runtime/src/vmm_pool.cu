#include "ggmlc/vmm_pool.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <cuda_runtime.h>
#include <cuda.h>

namespace ggmlc {

struct VMMBlockManager::Impl {
    int device_id = -1;
    CUdevice cu_device = 0;
    CUcontext cu_ctx = nullptr;
    size_t page_size = 0;
    bool initialized = false;

    // Accounting
    size_t total_reserved_va = 0;
    size_t total_mapped_bytes = 0;

    // Handles and Refcounts
    std::unordered_map<uint64_t, int> handle_refcounts;
    std::unordered_set<uint64_t> active_va_reservations;
    std::unordered_map<uint64_t, uint64_t> mapped_pages; // va_offset -> handle

    // Warm Pool & Upfront Allocation
    std::vector<uint64_t> free_pages_pool;
    size_t max_warm_pages = 64; // Default: 64 * 2MB = 128 MB warm pool
    size_t prealloc_pages = 0;

    // Prefix Caching
    std::unordered_map<std::string, uint64_t> prefix_page_table;

    CUmemAllocationProp alloc_prop = {};
    CUmemAccessDesc access_desc = {};
    std::mutex mtx;

    ~Impl() {
        reset();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!initialized) return;

        // 1. Clear prefix cache
        prefix_page_table.clear();

        // 2. Unmap all mapped pages
        for (const auto& pair : mapped_pages) {
            cuMemUnmap(static_cast<CUdeviceptr>(pair.first), page_size);
        }
        mapped_pages.clear();
        total_mapped_bytes = 0;

        // 3. Release warm pool handles
        for (uint64_t handle : free_pages_pool) {
            cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle));
        }
        free_pages_pool.clear();

        // 4. Release all physical handles
        for (const auto& pair : handle_refcounts) {
            CUmemGenericAllocationHandle handle = static_cast<CUmemGenericAllocationHandle>(pair.first);
            cuMemRelease(handle);
        }
        handle_refcounts.clear();

        // 4. Free all VA reservations
        for (uint64_t va : active_va_reservations) {
            // Note: size was tracked during reserve
            // In reset, reservations are cleared
        }
        active_va_reservations.clear();
        total_reserved_va = 0;
    }
};

VMMBlockManager::VMMBlockManager() : impl_(std::make_unique<Impl>()) {}
VMMBlockManager::~VMMBlockManager() = default;
VMMBlockManager::VMMBlockManager(VMMBlockManager&&) noexcept = default;
VMMBlockManager& VMMBlockManager::operator=(VMMBlockManager&&) noexcept = default;

bool VMMBlockManager::is_supported_on_device(int device_id) {
    if (cuInit(0) != CUDA_SUCCESS) return false;
    CUdevice dev;
    if (cuDeviceGet(&dev, device_id) != CUDA_SUCCESS) return false;
    int vmm_supported = 0;
    // CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
    if (cuDeviceGetAttribute(&vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev) != CUDA_SUCCESS) {
        return false;
    }
    return vmm_supported != 0;
}

bool VMMBlockManager::init(int device_id) {
    if (!impl_) impl_ = std::make_unique<Impl>();
    std::lock_guard<std::mutex> lock(impl_->mtx);

    if (impl_->initialized) return true;

    if (cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "[VMM] cuInit failed\n");
        return false;
    }

    if (cuDeviceGet(&impl_->cu_device, device_id) != CUDA_SUCCESS) {
        fprintf(stderr, "[VMM] cuDeviceGet failed for device %d\n", device_id);
        return false;
    }

    int vmm_supported = 0;
    if (cuDeviceGetAttribute(&vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, impl_->cu_device) != CUDA_SUCCESS || !vmm_supported) {
        fprintf(stderr, "[VMM] Device %d does not support Virtual Memory Management\n", device_id);
        return false;
    }

    if (cuDevicePrimaryCtxRetain(&impl_->cu_ctx, impl_->cu_device) != CUDA_SUCCESS) {
        fprintf(stderr, "[VMM] cuDevicePrimaryCtxRetain failed\n");
        return false;
    }
    cuCtxSetCurrent(impl_->cu_ctx);

    impl_->device_id = device_id;
    impl_->alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    impl_->alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    impl_->alloc_prop.location.id = device_id;

    impl_->access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    impl_->access_desc.location.id = device_id;
    impl_->access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    size_t gran = 0;
    if (cuMemGetAllocationGranularity(&gran, &impl_->alloc_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
        fprintf(stderr, "[VMM] cuMemGetAllocationGranularity failed\n");
        return false;
    }
    impl_->page_size = gran;
    impl_->initialized = true;

    return true;
}

bool VMMBlockManager::is_initialized() const {
    return impl_ && impl_->initialized;
}

size_t VMMBlockManager::page_size() const {
    return impl_ ? impl_->page_size : 0;
}

uint64_t VMMBlockManager::reserve_virtual_window(size_t window_bytes) {
    if (!impl_ || !impl_->initialized || window_bytes == 0) return 0;
    std::lock_guard<std::mutex> lock(impl_->mtx);

    // Round up window_bytes to page_size
    size_t aligned_size = ((window_bytes + impl_->page_size - 1) / impl_->page_size) * impl_->page_size;

    CUdeviceptr va_ptr = 0;
    CUresult res = cuMemAddressReserve(&va_ptr, aligned_size, 0, 0, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[VMM] cuMemAddressReserve failed for size %zu\n", aligned_size);
        return 0;
    }

    impl_->active_va_reservations.insert(static_cast<uint64_t>(va_ptr));
    impl_->total_reserved_va += aligned_size;
    return static_cast<uint64_t>(va_ptr);
}

void VMMBlockManager::free_virtual_window(uint64_t va_ptr, size_t window_bytes) {
    if (!impl_ || !impl_->initialized || va_ptr == 0) return;
    std::lock_guard<std::mutex> lock(impl_->mtx);

    size_t aligned_size = ((window_bytes + impl_->page_size - 1) / impl_->page_size) * impl_->page_size;
    cuMemAddressFree(static_cast<CUdeviceptr>(va_ptr), aligned_size);
    impl_->active_va_reservations.erase(va_ptr);
    if (impl_->total_reserved_va >= aligned_size) {
        impl_->total_reserved_va -= aligned_size;
    }
}

uint64_t VMMBlockManager::alloc_physical_page() {
    if (!impl_ || !impl_->initialized) return 0;
    std::lock_guard<std::mutex> lock(impl_->mtx);

    // 1. Fast-path: pop from warm free pool in O(1) without OS/driver syscall
    if (!impl_->free_pages_pool.empty()) {
        uint64_t handle = impl_->free_pages_pool.back();
        impl_->free_pages_pool.pop_back();
        impl_->handle_refcounts[handle] = 1;
        return handle;
    }

    // 2. Allocate fresh physical page from driver
    CUmemGenericAllocationHandle handle = 0;
    CUresult res = cuMemCreate(&handle, impl_->page_size, &impl_->alloc_prop, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[VMM] cuMemCreate failed\n");
        return 0;
    }

    impl_->handle_refcounts[static_cast<uint64_t>(handle)] = 1;
    return static_cast<uint64_t>(handle);
}

void VMMBlockManager::retain_physical_page(uint64_t page_handle) {
    if (!impl_ || page_handle == 0) return;
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->handle_refcounts.find(page_handle);
    if (it != impl_->handle_refcounts.end()) {
        it->second++;
    }
}

void VMMBlockManager::release_physical_page(uint64_t page_handle) {
    if (!impl_ || page_handle == 0) return;
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->handle_refcounts.find(page_handle);
    if (it == impl_->handle_refcounts.end()) return;

    it->second--;
    if (it->second <= 0) {
        impl_->handle_refcounts.erase(it);
        // Recycle handle into warm free pool if within configured capacity
        if (impl_->free_pages_pool.size() < impl_->max_warm_pages) {
            impl_->free_pages_pool.push_back(page_handle);
        } else {
            cuMemRelease(static_cast<CUmemGenericAllocationHandle>(page_handle));
        }
    }
}

bool VMMBlockManager::map_page(uint64_t va_offset, uint64_t page_handle) {
    if (!impl_ || !impl_->initialized || va_offset == 0 || page_handle == 0) return false;
    std::lock_guard<std::mutex> lock(impl_->mtx);

    // Map physical handle to virtual address
    CUresult res = cuMemMap(
        static_cast<CUdeviceptr>(va_offset),
        impl_->page_size,
        0,
        static_cast<CUmemGenericAllocationHandle>(page_handle),
        0
    );
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[VMM] cuMemMap failed at VA 0x%llx\n", (unsigned long long)va_offset);
        return false;
    }

    // Set Read-Write permissions
    res = cuMemSetAccess(static_cast<CUdeviceptr>(va_offset), impl_->page_size, &impl_->access_desc, 1);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[VMM] cuMemSetAccess failed at VA 0x%llx\n", (unsigned long long)va_offset);
        cuMemUnmap(static_cast<CUdeviceptr>(va_offset), impl_->page_size);
        return false;
    }

    impl_->mapped_pages[va_offset] = page_handle;
    impl_->total_mapped_bytes += impl_->page_size;
    return true;
}

bool VMMBlockManager::unmap_page(uint64_t va_offset) {
    if (!impl_ || !impl_->initialized || va_offset == 0) return false;
    std::lock_guard<std::mutex> lock(impl_->mtx);

    auto it = impl_->mapped_pages.find(va_offset);
    if (it == impl_->mapped_pages.end()) return false;

    CUresult res = cuMemUnmap(static_cast<CUdeviceptr>(va_offset), impl_->page_size);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[VMM] cuMemUnmap failed at VA 0x%llx\n", (unsigned long long)va_offset);
        return false;
    }

    impl_->mapped_pages.erase(it);
    if (impl_->total_mapped_bytes >= impl_->page_size) {
        impl_->total_mapped_bytes -= impl_->page_size;
    }
    return true;
}

uint64_t VMMBlockManager::get_prefix_page(const std::string& prefix_hash) {
    if (!impl_) return 0;
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->prefix_page_table.find(prefix_hash);
    if (it != impl_->prefix_page_table.end()) {
        uint64_t handle = it->second;
        impl_->handle_refcounts[handle]++;
        return handle;
    }
    return 0;
}

void VMMBlockManager::register_prefix_page(const std::string& prefix_hash, uint64_t page_handle) {
    if (!impl_ || page_handle == 0 || prefix_hash.empty()) return;
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->prefix_page_table[prefix_hash] = page_handle;
    impl_->handle_refcounts[page_handle]++;
}

void VMMBlockManager::evict_prefix_page(const std::string& prefix_hash) {
    if (!impl_) return;
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->prefix_page_table.find(prefix_hash);
    if (it != impl_->prefix_page_table.end()) {
        uint64_t handle = it->second;
        impl_->prefix_page_table.erase(it);
        auto ref_it = impl_->handle_refcounts.find(handle);
        if (ref_it != impl_->handle_refcounts.end()) {
            ref_it->second--;
            if (ref_it->second <= 0) {
                cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle));
                impl_->handle_refcounts.erase(ref_it);
            }
        }
    }
}

size_t VMMBlockManager::prefix_cache_size() const {
    return impl_ ? impl_->prefix_page_table.size() : 0;
}

size_t VMMBlockManager::total_reserved_va_bytes() const {
    return impl_ ? impl_->total_reserved_va : 0;
}

size_t VMMBlockManager::total_mapped_physical_bytes() const {
    return impl_ ? impl_->total_mapped_bytes : 0;
}

size_t VMMBlockManager::total_allocated_pages() const {
    if (!impl_) return 0;
    return impl_->handle_refcounts.size() + impl_->free_pages_pool.size();
}

void VMMBlockManager::configure_pool(size_t max_warm_pages, size_t prealloc_pages) {
    if (!impl_ || !impl_->initialized) return;
    std::lock_guard<std::mutex> lock(impl_->mtx);

    impl_->max_warm_pages = max_warm_pages;
    impl_->prealloc_pages = prealloc_pages;

    // Pre-allocate up to prealloc_pages if requested (e.g. --gpu-utilization)
    while (impl_->free_pages_pool.size() < prealloc_pages) {
        CUmemGenericAllocationHandle handle = 0;
        CUresult res = cuMemCreate(&handle, impl_->page_size, &impl_->alloc_prop, 0);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[VMM] cuMemCreate failed during upfront pre-allocation\n");
            break;
        }
        impl_->free_pages_pool.push_back(static_cast<uint64_t>(handle));
    }
}

size_t VMMBlockManager::warm_pool_pages() const {
    return impl_ ? impl_->free_pages_pool.size() : 0;
}

size_t VMMBlockManager::free_pool_pages() const {
    return impl_ ? impl_->free_pages_pool.size() : 0;
}

size_t VMMBlockManager::max_warm_pages() const {
    return impl_ ? impl_->max_warm_pages : 0;
}

void VMMBlockManager::drain_warm_pool() {
    if (!impl_) return;
    std::lock_guard<std::mutex> lock(impl_->mtx);
    for (uint64_t handle : impl_->free_pages_pool) {
        cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle));
    }
    impl_->free_pages_pool.clear();
}

void VMMBlockManager::reset() {
    if (impl_) {
        impl_->reset();
    }
}

} // namespace ggmlc
