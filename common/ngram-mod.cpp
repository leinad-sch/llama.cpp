#include "ngram-mod.h"

#include <algorithm>
#include <cstdio>
#include <string>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);
    scores.resize(size, SCORE_INIT);

    reset();
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    res = res % entries.size();

    return res;
}

void common_ngram_mod::add(const entry_t * tokens) {
    const size_t i = idx(tokens);

    if (entries[i] == EMPTY) {
        used++;
        scores[i] = SCORE_INS;
    } else if (entries[i] != tokens[n]) {
        // a different token hashes to the same bucket
        ++collisions;
    }
    // keep existing score if entry already occupied
    entries[i] = tokens[n];
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const size_t i = idx(tokens);

    return entries[i];
}

void common_ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), EMPTY);
    std::fill(scores.begin(), scores.end(), SCORE_INIT);
    used = 0;
    collisions = 0;
}

bool common_ngram_mod::save(const std::string & path) const {
    // format: magic(4) + version(4) + n(4) + size(4) + used(8) + entries(size*4)
    // magic: "NGRM" = 0x4E47524D
    const uint32_t magic = 0x4E47524D;
    const uint32_t version = 1;

    // write to a temp file for atomic rename
    const std::string tmp_path = path + ".tmp";
    FILE * f = fopen(tmp_path.c_str(), "wb");
    if (!f) {
        return false;
    }

    uint32_t n_u32 = (uint32_t)n;
    uint32_t sz_u32 = (uint32_t)entries.size();
    uint64_t used_u64 = (uint64_t)used;

    fwrite(&magic,   sizeof(magic),   1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&n_u32,   sizeof(n_u32),   1, f);
    fwrite(&sz_u32,  sizeof(sz_u32),  1, f);
    fwrite(&used_u64,sizeof(used_u64),1, f);
    fwrite(entries.data(), sizeof(entry_t), entries.size(), f);

    // flush and sync for crash safety
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

bool common_ngram_mod::load(const std::string & path) {
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

    // validate header
    if (magic != 0x4E47524D || version != 1 ||
        n_u32 != (uint32_t)n || sz_u32 != (uint32_t)entries.size())
    {
        fclose(f);
        return false;
    }

    // read entries directly into the vector
    if (fread(entries.data(), sizeof(entry_t), entries.size(), f) != entries.size()) {
        fclose(f);
        reset(); // restore clean state after partial write
        return false;
    }

    fclose(f);

    // recalculate used from the loaded data for robustness against file corruption
    used = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i] != EMPTY) {
            used++;
        }
    }

    return true;
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

size_t common_ngram_mod::size() const {
    return entries.size();
}

size_t common_ngram_mod::size_bytes() const {
    return entries.size() * sizeof(entries[0]) + scores.size() * sizeof(scores[0]);
}

size_t common_ngram_mod::index(const entry_t * tokens) const {
    return idx(tokens);
}

void common_ngram_mod::inc_score(const entry_t * tokens) {
    const size_t i = idx(tokens);
    if (scores[i] < common_ngram_mod::SCORE_MAX) {
        ++scores[i];
    }
}

void common_ngram_mod::dec_score(const entry_t * tokens) {
    const size_t i = idx(tokens);
    if (scores[i] > common_ngram_mod::SCORE_MIN) {
        --scores[i];
    }
}

void common_ngram_mod::inc_score_by_index(size_t i) {
    if (i < scores.size() && scores[i] < common_ngram_mod::SCORE_MAX) {
        ++scores[i];
    }
}

void common_ngram_mod::dec_score_by_index(size_t i) {
    if (i < scores.size() && scores[i] > common_ngram_mod::SCORE_MIN) {
        --scores[i];
    }
}

void common_ngram_mod::prune_low_score() {
    used = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i] != EMPTY) {
            if (scores[i] < common_ngram_mod::SCORE_THR) {
                entries[i] = EMPTY;
                scores[i] = 0;
            } else {
                ++used;
            }
        }
    }
}

size_t common_ngram_mod::get_collisions() const {
    return collisions;
}

size_t common_ngram_mod::get_below_thr() const {
    return count_below_thr;
}

size_t common_ngram_mod::get_at_min() const {
    return count_at_min;
}

size_t common_ngram_mod::get_at_max() const {
    return count_at_max;
}

size_t common_ngram_mod::get_at_ins() const {
    return count_at_ins;
}

void common_ngram_mod::update_score_stats() {
    // reset counters
    count_below_thr = 0;
    count_at_min   = 0;
    count_at_max   = 0;
    count_at_ins  = 0;

    for (size_t i = 0; i < scores.size(); ++i) {
        const int8_t s = scores[i];
        if (s < SCORE_THR) ++count_below_thr;
        if (s == SCORE_MIN) ++count_at_min;
        if (s == SCORE_MAX) ++count_at_max;
        if (s == SCORE_INS) ++count_at_ins;
    }
}
