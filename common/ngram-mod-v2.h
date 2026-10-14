#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

struct common_ngram_mod_v2 {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    common_ngram_mod_v2(uint16_t n, size_t size);

    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const;

    void clear();

    size_t get_n()        const;
    size_t get_used()     const;
    size_t size()         const;
    size_t size_bytes()   const;

    bool save(const std::string & path) const;
    bool load(const std::string & path);

private:
    struct slot {
        uint64_t key_hash = 0;
        entry_t  value    = EMPTY;
    };

    uint64_t hash_key(const entry_t * tokens) const;

    size_t n_;
    size_t used_ = 0;
    size_t max_used_;

    std::vector<slot> entries_;
};
