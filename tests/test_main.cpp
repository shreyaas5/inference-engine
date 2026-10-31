#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace inference_engine::testing {

bool run_loader_validation_tests();
bool run_quant_packing_tests();
bool run_kernels_scalar_tests();
bool run_kernels_simd_tests();
bool run_kv_cache_tests();
bool run_tokenizer_tests();
bool run_sampler_tests();
bool run_golden_forward_tests();
    bool run_cli_integration_test();

}  // namespace inference_engine::testing

int main(int argc, char** argv) {
    using namespace inference_engine::testing;

    struct Suite {
        std::string name;
        std::function<bool()> runner;
    };

    const std::vector<Suite> suites = {
        {"loader_validation", run_loader_validation_tests},
        {"quant_packing", run_quant_packing_tests},
        {"kernels_scalar", run_kernels_scalar_tests},
        {"kernels_simd", run_kernels_simd_tests},
        {"kv_cache", run_kv_cache_tests},
        {"tokenizer", run_tokenizer_tests},
        {"sampler", run_sampler_tests},
        {"cli_integration", run_cli_integration_test},
    };

    std::string filter;
    if (argc >= 3 && std::string(argv[1]) == "--suite") {
        filter = argv[2];
    }

    int passed = 0;
    int failed = 0;

    for (const auto& s : suites) {
        if (!filter.empty() && s.name != filter) {
            continue;
        }
        std::cout << "[ RUN      ] " << s.name << "\n";
        const bool ok = s.runner();
        if (ok) {
            std::cout << "[       OK ] " << s.name << "\n";
            ++passed;
        } else {
            std::cerr << "[  FAILED  ] " << s.name << "\n";
            ++failed;
        }
    }

    if (passed == 0 && failed == 0) {
        std::cerr << "No matching test suite found for filter: " << filter << "\n";
        return 1;
    }

    std::cout << "[==========] " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
