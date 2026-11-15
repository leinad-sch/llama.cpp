#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cstddef>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found

    void reset();

    // expose the hash index for external bookkeeping
    size_t index(const entry_t * tokens) const;

    // score handling
    void inc_score(const entry_t * tokens);
    void dec_score(const entry_t * tokens);
    void inc_score_by_index(size_t i);
    void dec_score_by_index(size_t i);
    void prune_low_score(); // remove entries below SCORE_THR

    // save/load the hash table to/from a file (atomic write on save)
    // returns true on success, false on error (load silently starts empty on failure)
    bool save(const std::string & path) const;
    bool load(const std::string & path);

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

private:
    size_t n; // ngram size to hash

    size_t used;

    std::vector<entry_t> entries;
};
