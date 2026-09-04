#include "ggml.h"
#include "llama.h"
#include "common.h"
#include "ngram-cache.h"

#include <clocale>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static void print_usage(char* argv0) {
    fprintf(stderr, "Merges multiple lookup cache files into a single one.\n");
    fprintf(stderr, "Usage: %s [--help] [--max-entries N] [--max-file-size-mib N] lookup_part_1.bin ... lookup_merged.bin\n", argv0);
}

int main(int argc, char ** argv){
    std::setlocale(LC_NUMERIC, "C");

    if (argc < 3) {
        print_usage(argv[0]);
        exit(1);
    }

    int32_t max_entries = 0;
    int32_t max_file_size_mib = 0;
    std::vector<std::string> args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--max-entries" && i + 1 < argc) {
            ++i;
            try {
                max_entries = std::stoi(argv[i]);
            } catch (const std::exception & e) {
                fprintf(stderr, "lookup-merge: invalid value for --max-entries: %s\n", argv[i]);
                exit(1);
            }
        } else if (arg == "--max-file-size-mib" && i + 1 < argc) {
            ++i;
            try {
                max_file_size_mib = std::stoi(argv[i]);
            } catch (const std::exception & e) {
                fprintf(stderr, "lookup-merge: invalid value for --max-file-size-mib: %s\n", argv[i]);
                exit(1);
            }
        } else {
            args.push_back(arg);
        }
    }

    if (args.size() < 2) {
        print_usage(argv[0]);
        exit(1);
    }

    fprintf(stderr, "lookup-merge: loading file %s\n", args[0].c_str());
    common_ngram_cache ngram_cache_merged = common_ngram_cache_load(args[0]);

    for (size_t i = 1; i < args.size()-1; ++i) {
        fprintf(stderr, "lookup-merge: loading file %s\n", args[i].c_str());
        common_ngram_cache ngram_cache = common_ngram_cache_load(args[i]);

        common_ngram_cache_merge(ngram_cache_merged, ngram_cache);
    }

    fprintf(stderr, "lookup-merge: saving file %s\n", args.back().c_str());
    common_ngram_cache_save(ngram_cache_merged, args.back(), max_entries, max_file_size_mib);
    return 0;
}
