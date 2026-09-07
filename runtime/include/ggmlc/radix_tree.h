#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <unordered_map>
#include <utility>

namespace ggmlc {

struct RadixNode {
    std::vector<int32_t> token_chunk;
    uint64_t chunk_hash = 0;
    
    // Physical page handles per attention layer (op_id -> page handle)
    std::unordered_map<uint32_t, uint64_t> k_handles;
    std::unordered_map<uint32_t, uint64_t> v_handles;

    std::weak_ptr<RadixNode> parent;
    std::unordered_map<uint64_t, std::shared_ptr<RadixNode>> children;

    int ref_count = 0;
    uint64_t last_accessed = 0;
};

struct PrefixMatchResult {
    size_t matched_tokens = 0;
    size_t matched_pages = 0;
    
    // Sequential physical handles per matched page
    std::vector<std::unordered_map<uint32_t, uint64_t>> page_k_handles;
    std::vector<std::unordered_map<uint32_t, uint64_t>> page_v_handles;
    std::vector<std::shared_ptr<RadixNode>> matched_nodes;
};

class PagedRadixTree {
public:
    explicit PagedRadixTree(size_t tokens_per_page = 1024);
    ~PagedRadixTree();

    PagedRadixTree(const PagedRadixTree&) = delete;
    PagedRadixTree& operator=(const PagedRadixTree&) = delete;

    // Match longest token sequence against tree in discrete tokens_per_page chunks
    PrefixMatchResult match_prefix(const std::vector<int32_t>& prompt_tokens);

    // Insert newly computed physical pages into the tree
    void insert_prefix(
        const std::vector<int32_t>& prompt_tokens,
        size_t start_page,
        const std::vector<std::unordered_map<uint32_t, uint64_t>>& page_k_handles,
        const std::vector<std::unordered_map<uint32_t, uint64_t>>& page_v_handles
    );

    // Reference counting for active request slots
    void retain_node(const std::shared_ptr<RadixNode>& node);
    void release_node(const std::shared_ptr<RadixNode>& node);

    // Evicts least-recently-used unreferenced nodes (ref_count == 0).
    // Returns list of evicted physical handles so VMM pool can recycle them.
    std::vector<uint64_t> evict_lru(size_t max_pages_to_evict);

    // Telemetry & metrics
    size_t total_cached_pages() const;
    size_t total_nodes() const;
    size_t tokens_per_page() const { return tokens_per_page_; }
    void reset();

    // 64-bit token chunk hasher (FNV-1a)
    static uint64_t hash_token_chunk(const int32_t* tokens, size_t count);

private:
    size_t tokens_per_page_;
    uint64_t current_tick_ = 0;
    std::shared_ptr<RadixNode> root_;

    void collect_eviction_candidates(
        const std::shared_ptr<RadixNode>& node,
        std::vector<std::shared_ptr<RadixNode>>& candidates
    );
    size_t count_pages_recursive(const std::shared_ptr<RadixNode>& node) const;
    size_t count_nodes_recursive(const std::shared_ptr<RadixNode>& node) const;
};

} // namespace ggmlc
