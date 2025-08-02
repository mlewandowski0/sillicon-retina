#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include "sillicon_retina_kernel.h"
#include "cnpy.h"

#define SAVE_ALL

using namespace std;

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                          \
    do                                                                            \
    {                                                                             \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess)                                                   \
        {                                                                         \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl;                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

struct SiliconRetinaConfig
{
    float alpha1 = 0.5f;
    float alpha2 = 0.5f;
    float V_BIP_LPF_alpha = 0.5f;
    float lambda_OPL = 3.f;
    float G_A_zero = 0.5f;
    float lambda_A = 0.5f;
    float p1 = 0.25f;
    float I_bias = 0.01f;
    float p2 = 1.f;
    float p3 = 0.2f;
    float U_rest = -0.02f;
    float U_th = 0.05f;
    float U_reset = 0.0f;

    void print() const
    {
        std::cout << "Silicon Retina Configuration:\n"
                  << "  alpha1: " << alpha1 << ", alpha2: " << alpha2
                  << ", V_BIP_LPF_alpha: " << V_BIP_LPF_alpha << "\n"
                  << "  lambda_OPL: " << lambda_OPL << ", G_A_zero: " << G_A_zero
                  << ", lambda_A: " << lambda_A << "\n"
                  << "  p1: " << p1 << ", I_bias: " << I_bias << ", p2: " << p2 << ", p3: " << p3 << "\n"
                  << "  U_rest: " << U_rest << ", U_th: " << U_th << ", U_reset: " << U_reset << std::endl;
    }
};

// RAII wrapper for CUDA memory management - FIXED VERSION
template <typename T>
class CudaMemory
{
private:
    T *d_ptr;
    size_t size_bytes;

public:
    explicit CudaMemory(size_t count) : size_bytes(count * sizeof(T))
    {
        CUDA_CHECK(cudaMalloc(&d_ptr, size_bytes));
    }

    ~CudaMemory()
    {
        if (d_ptr)
        {
            CUDA_CHECK(cudaFree(d_ptr));
        }
    }

    // Non-copyable but movable
    CudaMemory(const CudaMemory &) = delete;
    CudaMemory &operator=(const CudaMemory &) = delete;
    CudaMemory(CudaMemory &&other) noexcept : d_ptr(other.d_ptr), size_bytes(other.size_bytes)
    {
        other.d_ptr = nullptr;
        other.size_bytes = 0;
    }

    CudaMemory &operator=(CudaMemory &&other) noexcept
    {
        if (this != &other)
        {
            if (d_ptr)
                cudaFree(d_ptr);
            d_ptr = other.d_ptr;
            size_bytes = other.size_bytes;
            other.d_ptr = nullptr;
            other.size_bytes = 0;
        }
        return *this;
    }

    T *get() { return d_ptr; }
    const T *get() const { return d_ptr; }

    void copyFromHost(const std::vector<T> &host_data)
    {
        if (host_data.size() * sizeof(T) != size_bytes)
        {
            throw std::runtime_error("Size mismatch in copyFromHost");
        }
        CUDA_CHECK(cudaMemcpy(d_ptr, host_data.data(), size_bytes, cudaMemcpyHostToDevice));
    }

    void copyToHost(std::vector<T> &host_data) const // ← Made const
    {
        if (host_data.size() * sizeof(T) != size_bytes)
        {
            throw std::runtime_error("Size mismatch in copyToHost");
        }
        CUDA_CHECK(cudaMemcpy(host_data.data(), d_ptr, size_bytes, cudaMemcpyDeviceToHost));
    }

    void copyFromHostPartial(const T *host_data, size_t count, size_t offset = 0)
    {
        if ((count + offset) * sizeof(T) > size_bytes)
        {
            throw std::runtime_error("Size mismatch in copyFromHostPartial");
        }
        CUDA_CHECK(cudaMemcpy(d_ptr + offset, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copyToHostPartial(T *host_data, size_t count, size_t offset = 0) const // ← Made const
    {
        if ((count + offset) * sizeof(T) > size_bytes)
        {
            throw std::runtime_error("Size mismatch in copyToHostPartial");
        }
        CUDA_CHECK(cudaMemcpy(host_data, d_ptr + offset, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

std::vector<fs::path> list_files(const fs::path &directory)
{
    std::vector<fs::path> file_paths;

    if (!fs::exists(directory) || !fs::is_directory(directory))
    {
        throw std::runtime_error("Directory does not exist or is not a directory: " + directory.string());
    }

    for (auto const &entry : fs::directory_iterator(directory))
    {
        if (fs::is_regular_file(entry.status()) && entry.path().extension() == ".png")
        {
            file_paths.push_back(entry.path());
        }
    }

    return file_paths;
}

// Load a single image and return as float vector
std::vector<float> load_single_image(const fs::path &file_path, int &W, int &H, bool is_first = false)
{
    cv::Mat img8 = cv::imread(file_path.string(), cv::IMREAD_GRAYSCALE);

    if (img8.empty())
    {
        throw std::runtime_error("Failed to load image: " + file_path.string());
    }

    if (is_first)
    {
        W = img8.cols;
        H = img8.rows;
        std::cout << "Image dimensions: " << W << "x" << H << std::endl;
    }
    else
    {
        // Validate dimensions consistency
        if (img8.cols != W || img8.rows != H)
        {
            throw std::runtime_error("Image dimension mismatch in " + file_path.string() +
                                     ": expected " + std::to_string(W) + "x" + std::to_string(H) +
                                     ", got " + std::to_string(img8.cols) + "x" + std::to_string(img8.rows));
        }
    }

    // Convert to float and normalize
    cv::Mat imgf;
    img8.convertTo(imgf, CV_32F, 1.0f / 255.0f);

    // Copy to vector
    std::vector<float> result(W * H);
    std::memcpy(result.data(), imgf.ptr<float>(), sizeof(float) * W * H);

    return result;
}

void ensure_directory_exists(const std::string &path)
{
    if (!fs::exists(path))
    {
        fs::create_directories(path);
        std::cout << "Created directory: " << path << std::endl;
    }
}

// Save a single frame (for current buffer state)
void save_current_frame_state(int frame_idx, int H, int W,
                              const CudaMemory<float> &d_LPF1, const CudaMemory<float> &d_HPF1,
                              const CudaMemory<float> &d_BLURRED2, const CudaMemory<float> &d_LPF2,
                              const CudaMemory<float> &d_I_OPL, const CudaMemory<float> &d_V_BIP,
                              const CudaMemory<float> &d_Q_V_BIP, const CudaMemory<float> &d_V_BIP_LPF,
                              const CudaMemory<float> &d_HPF2, const CudaMemory<float> &d_I_Gang,
                              const CudaMemory<float> &d_V_LIF, const CudaMemory<unsigned char> &d_fired,
                              int current_buffer_idx, bool save_intermediate)
{
    const size_t frame_size = H * W;
    const size_t buffer_offset = current_buffer_idx * frame_size;

    // Always save fired events (from the double buffer)
    std::vector<unsigned char> fired_frame(frame_size);
    d_fired.copyToHostPartial(fired_frame.data(), frame_size, 0);

    std::string out_path = "out/fired/" + std::to_string(frame_idx) + ".npy";
    cnpy::npy_save(out_path, fired_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");

    // Always save V_LIF (membrane potential) - this is often the most important state
    std::vector<float> vlif_frame(frame_size);
    d_V_LIF.copyToHostPartial(vlif_frame.data(), frame_size, buffer_offset);

    out_path = "out/VLIF/" + std::to_string(frame_idx) + ".npy";
    cnpy::npy_save(out_path, vlif_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");

#ifdef SAVE_ALL
    if (save_intermediate)
    {
        // Save intermediate results from current buffer
        std::vector<float> temp_frame(frame_size);

        // LPF1
        d_LPF1.copyToHostPartial(temp_frame.data(), frame_size, buffer_offset);
        out_path = "out/LPF1/" + std::to_string(frame_idx) + ".npy";
        cnpy::npy_save(out_path, temp_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");

        // HPF1
        d_HPF1.copyToHostPartial(temp_frame.data(), frame_size, buffer_offset);
        out_path = "out/HPF1/" + std::to_string(frame_idx) + ".npy";
        cnpy::npy_save(out_path, temp_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");

        // LPF2
        d_LPF2.copyToHostPartial(temp_frame.data(), frame_size, buffer_offset);
        out_path = "out/LPF2/" + std::to_string(frame_idx) + ".npy";
        cnpy::npy_save(out_path, temp_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");

        // BLURRED2
        d_BLURRED2.copyToHostPartial(temp_frame.data(), frame_size, buffer_offset);
        out_path = "out/BLUR2/" + std::to_string(frame_idx) + ".npy";
        cnpy::npy_save(out_path, temp_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");

        // I_OPL
        d_I_OPL.copyToHostPartial(temp_frame.data(), frame_size, buffer_offset);
        out_path = "out/I_OPL/" + std::to_string(frame_idx) + ".npy";
        cnpy::npy_save(out_path, temp_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");

        // V_BIP
        d_V_BIP.copyToHostPartial(temp_frame.data(), frame_size, buffer_offset);
        out_path = "out/VBIP/" + std::to_string(frame_idx) + ".npy";
        cnpy::npy_save(out_path, temp_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");

        // Q_V_BIP
        d_Q_V_BIP.copyToHostPartial(temp_frame.data(), frame_size, buffer_offset);
        out_path = "out/QVBIP/" + std::to_string(frame_idx) + ".npy";
        cnpy::npy_save(out_path, temp_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");

        // V_BIP_LPF
        d_V_BIP_LPF.copyToHostPartial(temp_frame.data(), frame_size, buffer_offset);
        out_path = "out/VBIP_LPF/" + std::to_string(frame_idx) + ".npy";
        cnpy::npy_save(out_path, temp_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");

        // HPF2
        d_HPF2.copyToHostPartial(temp_frame.data(), frame_size, buffer_offset);
        out_path = "out/HPF2/" + std::to_string(frame_idx) + ".npy";
        cnpy::npy_save(out_path, temp_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");

        // I_Gang
        d_I_Gang.copyToHostPartial(temp_frame.data(), frame_size, buffer_offset);
        out_path = "out/IGang/" + std::to_string(frame_idx) + ".npy";
        cnpy::npy_save(out_path, temp_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");

        // Note: V_LIF is already saved above (always saved, not just with --save-intermediate)
    }
#endif
}

// Online processing function - processes images one by one
double run_silicon_retina_online_processing(const SiliconRetinaConfig &config,
                                            const std::vector<fs::path> &files,
                                            int H, int W, bool save_intermediate)
{
    const int T = static_cast<int>(files.size());
    const size_t frame_size = H * W;
    const size_t double_buffer_size = 2 * frame_size; // Double buffering for state

    std::cout << "=== Online Processing Mode ===" << std::endl;
    std::cout << "Processing " << T << " frames with dimensions " << W << "x" << H << std::endl;
    std::cout << "Memory usage: " << (double_buffer_size * 11 * sizeof(float) + double_buffer_size * sizeof(unsigned char)) / (1024.0 * 1024.0) << " MB" << std::endl;

    // Allocate device memory for double buffering (2 frames worth)
    CudaMemory<float> d_current_img(frame_size);  // Current input frame
    CudaMemory<float> d_LPF1(double_buffer_size); // Double buffered state arrays
    CudaMemory<float> d_HPF1(double_buffer_size);
    CudaMemory<float> d_BLURRED2(double_buffer_size);
    CudaMemory<float> d_LPF2(double_buffer_size);
    CudaMemory<float> d_I_OPL(double_buffer_size);
    CudaMemory<float> d_V_BIP(double_buffer_size);
    CudaMemory<float> d_Q_V_BIP(double_buffer_size);
    CudaMemory<float> d_V_BIP_LPF(double_buffer_size);
    CudaMemory<float> d_HPF2(double_buffer_size);
    CudaMemory<float> d_I_Gang(double_buffer_size);
    CudaMemory<float> d_V_LIF(double_buffer_size);
    CudaMemory<unsigned char> d_fired(frame_size);

    // Initialize all arrays to zero
    std::vector<float> zero_buffer_float(double_buffer_size, 0.0f);
    std::vector<unsigned char> zero_buffer_char(frame_size, 0);

    d_LPF1.copyFromHost(zero_buffer_float);
    d_HPF1.copyFromHost(zero_buffer_float);
    d_BLURRED2.copyFromHost(zero_buffer_float);
    d_LPF2.copyFromHost(zero_buffer_float);
    d_I_OPL.copyFromHost(zero_buffer_float);
    d_V_BIP.copyFromHost(zero_buffer_float);
    d_Q_V_BIP.copyFromHost(zero_buffer_float);
    d_V_BIP_LPF.copyFromHost(zero_buffer_float);
    d_HPF2.copyFromHost(zero_buffer_float);
    d_I_Gang.copyFromHost(zero_buffer_float);
    d_V_LIF.copyFromHost(zero_buffer_float);
    d_fired.copyFromHost(zero_buffer_char);

    // Setup output directories
    ensure_directory_exists("out");
    ensure_directory_exists("out/fired");
    ensure_directory_exists("out/VLIF"); // Always create VLIF directory since we always save it

    if (save_intermediate)
    {
        const std::vector<std::string> output_dirs = {
            "out/imgs", "out/LPF1", "out/LPF2", "out/BLUR2", "out/HPF1",
            "out/I_OPL", "out/VBIP", "out/QVBIP", "out/VBIP_LPF", "out/HPF2",
            "out/IGang"};
        for (const auto &dir : output_dirs)
        {
            ensure_directory_exists(dir);
        }
    }

    double total_ms = 0;

        // Process each frame
        for (int curr_t = 0; curr_t < T; ++curr_t)
    {
        // Load current frame
        int dummy_W, dummy_H;
        auto current_frame = load_single_image(files[curr_t], dummy_W, dummy_H, curr_t == 0);
        auto processing_start = std::chrono::high_resolution_clock::now();

        // Upload current frame to GPU
        d_current_img.copyFromHost(current_frame);

        // Save input image if requested
        if (save_intermediate)
        {
            std::string out_path = "out/imgs/" + std::to_string(curr_t) + ".npy";
            cnpy::npy_save(out_path, current_frame.data(), {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");
        }

        // Run silicon retina kernel for this time step
        run_silicon_retina_model_online(
            d_current_img.get(),
            d_LPF1.get(),
            d_HPF1.get(),
            d_BLURRED2.get(),
            d_LPF2.get(),
            d_I_OPL.get(),
            d_V_BIP.get(),
            d_Q_V_BIP.get(),
            d_V_BIP_LPF.get(),
            d_HPF2.get(),
            d_I_Gang.get(),
            d_V_LIF.get(),
            d_fired.get(),
            curr_t + 1, H, W,
            config.alpha1,
            config.alpha2,
            config.V_BIP_LPF_alpha,
            config.lambda_OPL,
            config.G_A_zero,
            config.lambda_A,
            config.p1,
            config.I_bias,
            config.p2,
            config.p3,
            config.U_rest,
            config.U_th,
            config.U_reset);

        // Synchronize to ensure kernel completion
        CUDA_CHECK(cudaDeviceSynchronize());

        auto processing_end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(processing_end - processing_start).count();

        total_ms += duration_ms;

        // Save results for this frame (only if not initialization frame)
        if (curr_t > 0)
        {
            int current_buffer_idx = curr_t % 2;
            save_current_frame_state(curr_t, H, W,
                                     d_LPF1, d_HPF1, d_BLURRED2, d_LPF2, d_I_OPL,
                                     d_V_BIP, d_Q_V_BIP, d_V_BIP_LPF, d_HPF2,
                                     d_I_Gang, d_V_LIF, d_fired,
                                     current_buffer_idx, save_intermediate);
        }

        // Progress report
        if ((curr_t + 1) % 10 == 0 || curr_t == T - 1)
        {
            std::cout << "Processed " << (curr_t + 1) << "/" << T << " frames" << std::endl;
        }
    }

    return total_ms;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <image_directory> [--save-intermediate]\n"
                  << "  image_directory: Path to directory containing image sequence\n"
                  << "  --save-intermediate: Save all intermediate processing states (optional)\n"
                  << std::endl;
        return 1;
    }

    fs::path cwd = fs::current_path();
    fs::path directory_path = argv[1];
    bool save_intermediate = false;

    // Check for optional flag
    if (argc > 2 && std::string(argv[2]) == "--save-intermediate")
    {
        save_intermediate = true;
    }

    std::cout << "=== Online Silicon Retina Processing ===" << std::endl;
    std::cout << "Current working directory: " << cwd << std::endl;
    std::cout << "Input directory: " << directory_path << std::endl;
    std::cout << "Save intermediate results: " << (save_intermediate ? "Yes" : "No") << std::endl;

    // Configuration
    SiliconRetinaConfig config;
    config.print();

    // Load file list
    std::vector<fs::path> files;
    try
    {
        files = list_files(directory_path);
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    if (files.empty())
    {
        std::cout << "No PNG files found in the directory: " << directory_path << std::endl;
        return 1;
    }

    std::sort(files.begin(), files.end());
    std::cout << "Found " << files.size() << " PNG files" << std::endl;

    // Get dimensions from first image
    int W = 0, H = 0;
    try
    {
        auto first_frame = load_single_image(files[0], W, H, true);
        std::cout << "Loaded first image with dimensions: " << W << "x" << H << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading first image: " << e.what() << std::endl;
        return 1;
    }

    // Run online processing
    try
    {
        double duration_ms = run_silicon_retina_online_processing(config, files, H, W, save_intermediate);

        const int T = static_cast<int>(files.size());
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n=== Processing Results ===" << std::endl;
        std::cout << "Total processing time: " << duration_ms << " ms" << std::endl;
        std::cout << "Average frame time: " << duration_ms / T << " ms" << std::endl;
        std::cout << "Throughput: " << (T * 1000.0 / duration_ms) << " frames/second" << std::endl;

        // Calculate memory savings compared to traditional approach
        size_t online_memory = 2 * H * W * 11 * sizeof(float) + 2 * H * W * sizeof(unsigned char);
        size_t traditional_memory = T * H * W * 11 * sizeof(float) + T * H * W * sizeof(unsigned char);
        double memory_reduction = (1.0 - static_cast<double>(online_memory) / traditional_memory) * 100.0;

        std::cout << "\n=== Memory Usage ===" << std::endl;
        std::cout << "Online approach: " << online_memory / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Traditional approach: " << traditional_memory / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Memory reduction: " << memory_reduction << "%" << std::endl;

        std::cout << "\n=== Processing Completed Successfully ===" << std::endl;
        std::cout << "Results saved to: out/" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}