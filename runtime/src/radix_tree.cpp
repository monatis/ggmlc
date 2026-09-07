#include "ggmlc/radix_tree.h"
#include <algorithm>
#include <cstring>

namespace ggmlc {

uint64_t PagedRadixTree::hash_token_chunk(const int32_t* tokens, size_t count) {
    // 64-bit FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < count; ++i) {
        uint32_t val = static_cast<uint32_t>(tokens[i]);
        hash ^= (val & 0xFF);
        hash *= 1099511628211ULL;
        hash ^= ((val >> 8) & 0xFF);
        hash *= 1099511628211ULL;
        hash ^= ((val >> 16) & 0xFF);
        hash *= 1099511628211ULL;
        hash ^= ((val >> 24) & 0xFF);
        hash *= 1099511628211ULL;
    }
    return hash;
}

PagedRadixTree::PagedRadixTree(size_t tokens_per_page)
    : tokens_per_page_(tokens_per_page > 0 ? tokens_per_page : 1024) {
    root_ = std::make_shared<RadixNode>();
}

PagedRadixTree::~PagedRadixTree() = default;

PrefixMatchResult PagedRadixTree::match_prefix(const std::vector<int32_t>& prompt_tokens) {
    PrefixMatchResult result;
    if (tokens_per_page_ == 0 || prompt_tokens.size() < tokens_per_page_) {
        return result;
    }

    size_t num_full_pages = prompt_tokens.size() / tokens_per_page_;
    std::shared_ptr<RadixNode> current = root_;

    for (size_t p = 0; p < num_full_pages; ++p) {
        const int32_t* chunk_ptr = prompt_tokens.data() + p * tokens_per_page_;
        uint64_t chunk_h = hash_token_chunk(chunk_ptr, tokens_per_page_);

        auto it = current->children.find(chunk_h);
        if (it == current->children.end()) {
            break;
        }

        const auto& candidate = it->second;
        // Verify exact token equality to guard against any hash collision
        if (candidate->token_chunk.size() != tokens_per_page_ ||
            std::memcmp(candidate->token_chunk.data(), chunk_ptr, tokens_per_page_ * sizeof(int32_t)) != 0) {
            break;
        }

        current = candidate;
        current->last_accessed = ++current_tick_;

        result.matched_nodes.push_back(current);
        result.page_k_handles.push_back(current->k_handles);
        result.page_v_handles.push_back(current->v_handles);
        result.matched_pages++;
        result.matched_tokens += tokens_per_page_;
    }

    return result;
}

void PagedRadixTree::insert_prefix(
    const std::vector<int32_t>& prompt_tokens,
    size_t start_page,
    const std::vector<std::unordered_map<uint32_t, uint64_t>>& page_k_handles,
    const std::vector<std::unordered_map<uint32_t, uint64_t>>& page_v_handles
) {
    if (tokens_per_page_ == 0 || page_k_handles.empty() || page_k_handles.size() != page_v_handles.size()) {
        return;
    }

    // Traverse down the existing prefix up to start_page
    std::shared_ptr<RadixNode> current = root_;
    for (size_t p = 0; p < start_page; ++p) {
        if ((p + 1) * tokens_per_page_ > prompt_tokens.size()) return;
        const int32_t* chunk_ptr = prompt_tokens.data() + p * tokens_per_page_;
        uint64_t chunk_h = hash_token_chunk(chunk_ptr, tokens_per_page_);
        auto it = current->children.find(chunk_h);
        if (it == current->children.end()) {
            // Prefix broke early; cannot attach
            return;
        }
        current = it->second;
    }

    // Insert subsequent newly computed pages
    for (size_t j = 0; j < page_k_handles.size(); ++j) {
        size_t page_idx = start_page + j;
        size_t chunk_start = page_idx * tokens_per_page_;
        if (chunk_start + tokens_per_page_ > prompt_tokens.size()) {
            break; // Fractional tail cannot form an immutable page
        }

        const int32_t* chunk_ptr = prompt_tokens.data() + chunk_start;
        uint64_t chunk_h = hash_token_chunk(chunk_ptr, tokens_per_page_);

        auto it = current->children.find(chunk_h);
        if (it != current->children.end()) {
            current = it->second;
            current->last_accessed = ++current_tick_;
        } else {
            auto new_node = std::make_shared<RadixNode>();
            new_node->token_chunk.assign(chunk_ptr, chunk_ptr + tokens_per_page_);
            new_node->chunk_hash = chunk_h;
            new_node->k_handles = page_k_handles[j];
            new_node->v_handles = page_v_handles[j];
            new_node->parent = current;
            new_node->ref_count = 0;
            new_node->last_accessed = ++current_tick_;

            current->children[chunk_h] = new_node;
            current = new_node;
        }
    }
}

void PagedRadixTree::retain_node(const std::shared_ptr<RadixNode>& node) {
    if (node) {
        node->ref_count++;
        node->last_accessed = ++current_tick_;
    }
}

void PagedRadixTree::release_node(const std::shared_ptr<RadixNode>& node) {
    if (node && node->ref_count > 0) {
        node->ref_count--;
    }
}

void PagedRadixTree::collect_eviction_candidates(
    const std::shared_ptr<RadixNode>& node,
    std::vector<std::shared_ptr<RadixNode>>& candidates
) {
    if (!node) return;
    for (const auto& pair : node->children) {
        const auto& child = pair.second;
        if (child->children.empty() && child->ref_count == 0) {
            candidates.push_back(child);
        } else {
            collect_eviction_candidates(child, candidates);
        }
    }
}

std::vector<uint64_t> PagedRadixTree::evict_lru(size_t max_pages_to_evict) {
    std::vector<uint64_t> freed_handles;
    if (max_pages_to_evict == 0) return freed_handles;

    size_t evicted_count = 0;
    while (evicted_count < max_pages_to_evict) {
        std::vector<std::shared_ptr<RadixNode>> leaf_candidates;
        collect_eviction_candidates(root_, leaf_candidates);
        if (leaf_candidates.empty()) break;

        // Sort by last_accessed ascending (oldest first)
        std::sort(leaf_candidates.begin(), leaf_candidates.end(),
            [](const std::shared_ptr<RadixNode>& a, const std::shared_ptr<RadixNode>& b) {
                return a->last_accessed < b->last_accessed;
            });

        auto target = leaf_candidates.front();
        for (const auto& pair : target->k_handles) {
            freed_handles.push_back(pair.second);
        }
        for (const auto& pair : target->v_handles) {
            freed_handles.push_back(pair.second);
        }

        // Unlink from parent
        auto parent_sp = target->parent.lock();
        if (parent_sp) {
            parent_sp->children.erase(target->chunk_hash);
        }
        evicted_count++;
    }

    return freed_handles;
}

size_t PagedRadixTree::count_pages_recursive(const std::shared_ptr<RadixNode>& node) const {
    if (!node) return 0;
    size_t count = (node != root_) ? 1 : 0;
    for (const auto& pair : node->children) {
        count += count_pages_recursive(pair.second);
    }
    return count;
}

size_t PagedRadixTree::count_nodes_recursive(const std::shared_ptr<RadixNode>& node) const {
    if (!node) return 0;
    size_t count = 1;
    for (const auto& pair : node->children) {
        count += count_nodes_recursive(pair.second);
    }
    return count;
}

size_t PagedRadixTree::total_cached_pages() const {
    return count_pages_recursive(root_);
}

size_t PagedRadixTree::total_nodes() const {
    return count_nodes_recursive(root_);
}

void PagedRadixTree::reset() {
    if (root_) {
        root_->children.clear();
    }
    current_tick_ = 0;
}

} // namespace ggmlc
