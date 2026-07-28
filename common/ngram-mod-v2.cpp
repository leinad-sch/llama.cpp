#include "ngram-mod-v2.h"
#include "log.h"

#include <algorithm>
#include <cstdio>
#include <string>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

common_ngram_mod_v2::common_ngram_mod_v2(uint16_t n, size_t size)
    : n_(n)
    , entries_(size)
{
    max_used_ = entries_.size() * 9 / 10;
}

uint64_t common_ngram_mod_v2::hash_key(const entry_t * tokens) const {
    uint64_t h = 0;

    for (size_t i = 0; i < n_; ++i) {
        h = h * 6364136223846793005ULL + (uint64_t)(int64_t)tokens[i];
    }

    return h ? h : 1;
}

void common_ngram_mod_v2::add(const entry_t * tokens) {
    if (used_ >= max_used_) {
        LOG_WRN("%s: hash table %zu/%zu full, resetting\n", __func__, used_, entries_.size());
        clear();
    }

    const uint64_t h = hash_key(tokens);
    size_t i = h % entries_.size();

    while (entries_[i].key_hash != 0 && entries_[i].key_hash != h) {
        if (++i >= entries_.size()) {
            i = 0;
        }
    }

    if (entries_[i].key_hash == 0) {
        used_++;
    }

    entries_[i].key_hash = h;
    entries_[i].value    = tokens[n_];
}

common_ngram_mod_v2::entry_t common_ngram_mod_v2::get(const entry_t * tokens) const {
    const uint64_t h = hash_key(tokens);
    size_t i = h % entries_.size();

    while (entries_[i].key_hash != 0) {
        if (entries_[i].key_hash == h) {
            return entries_[i].value;
        }

        if (++i >= entries_.size()) {
            i = 0;
        }
    }

    return EMPTY;
}

void common_ngram_mod_v2::clear() {
    std::fill(entries_.begin(), entries_.end(), slot{});
    used_ = 0;
}

size_t common_ngram_mod_v2::get_n() const {
    return n_;
}

size_t common_ngram_mod_v2::get_used() const {
    return used_;
}

size_t common_ngram_mod_v2::size() const {
    return entries_.size();
}

size_t common_ngram_mod_v2::size_bytes() const {
    return entries_.size() * sizeof(entries_[0]);
}

bool common_ngram_mod_v2::save(const std::string & path) const {
    const uint32_t magic   = 0x4E475232; // "NGR2"
    const uint32_t version = 1;

    const std::string tmp_path = path + ".tmp";
    FILE * f = fopen(tmp_path.c_str(), "wb");
    if (!f) {
        return false;
    }

    const uint32_t n_u32   = (uint32_t)n_;
    const uint32_t sz_u32  = (uint32_t)entries_.size();
    const uint64_t used_u64 = (uint64_t)used_;

    fwrite(&magic,   sizeof(magic),   1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&n_u32,   sizeof(n_u32),   1, f);
    fwrite(&sz_u32,  sizeof(sz_u32),  1, f);
    fwrite(&used_u64,sizeof(used_u64),1, f);

    for (size_t i = 0; i < entries_.size(); ++i) {
        fwrite(&entries_[i].key_hash, sizeof(entries_[i].key_hash), 1, f);
        fwrite(&entries_[i].value,    sizeof(entries_[i].value),    1, f);
    }

    if (fflush(f) != 0) {
        fclose(f);
        remove(tmp_path.c_str());
        return false;
    }
#if defined(_WIN32)
    if (_commit(_fileno(f)) != 0) {
        fclose(f);
        remove(tmp_path.c_str());
        return false;
    }
#else
    if (fsync(fileno(f)) != 0) {
        fclose(f);
        remove(tmp_path.c_str());
        return false;
    }
#endif
    if (fclose(f) != 0) {
        remove(tmp_path.c_str());
        return false;
    }

    if (rename(tmp_path.c_str(), path.c_str()) != 0) {
        remove(tmp_path.c_str());
        return false;
    }

    return true;
}

bool common_ngram_mod_v2::load(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }

    uint32_t magic;
    uint32_t version;
    uint32_t n_u32;
    uint32_t sz_u32;
    uint64_t used_u64;

    if (fread(&magic,   sizeof(magic),   1, f) != 1 ||
        fread(&version, sizeof(version), 1, f) != 1 ||
        fread(&n_u32,   sizeof(n_u32),   1, f) != 1 ||
        fread(&sz_u32,  sizeof(sz_u32),  1, f) != 1 ||
        fread(&used_u64,sizeof(used_u64),1, f) != 1)
    {
        fclose(f);
        return false;
    }

    if (magic != 0x4E475232 || version != 1 ||
        n_u32 != (uint32_t)n_ || sz_u32 != (uint32_t)entries_.size())
    {
        fclose(f);
        return false;
    }

    clear();

    for (size_t i = 0; i < entries_.size(); ++i) {
        if (fread(&entries_[i].key_hash, sizeof(entries_[i].key_hash), 1, f) != 1 ||
            fread(&entries_[i].value,    sizeof(entries_[i].value),    1, f) != 1)
        {
            clear();
            fclose(f);
            return false;
        }
    }

    fclose(f);

    used_ = 0;
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].key_hash != 0) {
            used_++;
        }
    }

    return true;
}
