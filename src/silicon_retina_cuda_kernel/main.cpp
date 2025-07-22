#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include "sillicon_retina_kernel.h"
#include "cnpy.h"

#define SAVE_TO_FOLDER
#define SAVE_ALL

using namespace std;
namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                          \
\
    do                                                                            \
    {                                                                             \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess)                                                   \
        {                                                                         \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl;                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    }                                                                             \
    while (0)

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

    std::string as_string() const
    {
        return "SiliconRetinaConfig(alpha1=" + std::to_string(alpha1) +
               ", alpha2=" + std::to_string(alpha2) +
               ", V_BIP_LPF_alpha=" + std::to_string(V_BIP_LPF_alpha) +
               ", lambda_OPL=" + std::to_string(lambda_OPL) +
               ", G_A_zero=" + std::to_string(G_A_zero) +
               ", lambda_A=" + std::to_string(lambda_A) +
               ", p1=" + std::to_string(p1) +
               ", I_bias=" + std::to_string(I_bias) +
               ", p2=" + std::to_string(p2) +
               ", p3=" + std::to_string(p3) +
               ", U_rest=" + std::to_string(U_rest) +
               ", U_th=" + std::to_string(U_th) +
               ", U_reset=" + std::to_string(U_reset) + ")";
    }

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

// RAII wrapper for CUDA memory management
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

    void copyToHost(std::vector<T> &host_data)
    {
        if (host_data.size() * sizeof(T) != size_bytes)
        {
            throw std::runtime_error("Size mismatch in copyToHost");
        }
        CUDA_CHECK(cudaMemcpy(host_data.data(), d_ptr, size_bytes, cudaMemcpyDeviceToHost));
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

template <typename T>
void save_frame_generic(const std::vector<T> &data,
                        int frame_idx,
                        int Tsize, int H, int W,
                        const std::string &out_path,
                        bool save_as_npy = true)
{
    // sanity checks
    if (frame_idx < 0 || frame_idx >= Tsize)
    {
        throw std::out_of_range("frame_idx out of bounds");
    }

    if ((int)data.size() != Tsize * H * W)
    {
        throw std::runtime_error("data.size() doesn't match T*H*W");
    }

    const T *ptr = data.data() + frame_idx * H * W;

    if (save_as_npy)
    {
        cnpy::npy_save(out_path, ptr, {static_cast<size_t>(H), static_cast<size_t>(W)}, "w");
    }
    else
    {
        // Handle different data types for OpenCV
        cv::Mat img;
        if constexpr (std::is_same_v<T, unsigned char>)
        {
            img = cv::Mat(H, W, CV_8U, const_cast<T *>(ptr));
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            img = cv::Mat(H, W, CV_32F, const_cast<T *>(ptr));
            cv::Mat img_u8;
            img.convertTo(img_u8, CV_8U, 255.0, 0.0);
            img = img_u8;
        }

        if (!cv::imwrite(out_path, img))
        {
            throw std::runtime_error("Failed to save image to " + out_path);
        }
    }
}

void ensure_directory_exists(const std::string &path)
{
    if (!fs::exists(path))
    {
        fs::create_directories(path);
        std::cout << "Created directory: " << path << std::endl;
    }
}

double run_silicon_retina_processing(const SiliconRetinaConfig &config,
                                     int mode, int T, int H, int W,
                                     CudaMemory<float> &d_imgs,
                                     CudaMemory<float> &d_LPF1, CudaMemory<float> &d_HPF1, CudaMemory<float> &d_BLURRED2,
                                     CudaMemory<float> &d_LPF2, CudaMemory<float> &d_I_OPL, CudaMemory<float> &d_V_BIP,
                                     CudaMemory<float> &d_Q_V_BIP, CudaMemory<float> &d_V_BIP_LPF, CudaMemory<float> &d_HPF2,
                                     CudaMemory<float> &d_I_Gang, CudaMemory<float> &d_V_LIF, CudaMemory<unsigned char> &d_fired)
{
    auto begin = std::chrono::high_resolution_clock::now();

    if (mode == 0)
    {
        run_silicon_retina_model(
            d_imgs.get(), d_LPF1.get(), d_HPF1.get(), d_BLURRED2.get(), d_LPF2.get(),
            d_I_OPL.get(), d_V_BIP.get(), d_Q_V_BIP.get(), d_V_BIP_LPF.get(),
            d_HPF2.get(), d_I_Gang.get(), d_V_LIF.get(), d_fired.get(),
            T, H, W, config.alpha1, config.alpha2, config.V_BIP_LPF_alpha,
            config.lambda_OPL, config.G_A_zero, config.lambda_A, config.p1,
            config.I_bias, config.p2, config.p3, config.U_rest, config.U_th, config.U_reset);
    }
    else
    {
        run_silicon_retina_model_fired_only(
            d_imgs.get(), d_LPF1.get(), d_HPF1.get(), d_BLURRED2.get(), d_LPF2.get(),
            d_I_OPL.get(), d_V_BIP.get(), d_Q_V_BIP.get(), d_V_BIP_LPF.get(),
            d_HPF2.get(), d_I_Gang.get(), d_V_LIF.get(), d_fired.get(),
            T, H, W, config.alpha1, config.alpha2, config.V_BIP_LPF_alpha,
            config.lambda_OPL, config.G_A_zero, config.lambda_A, config.p1,
            config.I_bias, config.p2, config.p3, config.U_rest, config.U_th, config.U_reset);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double duration_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    return duration_ms;
}

void save_results(int mode, int T, int H, int W,
                  const std::vector<float> &imgs,
                  const std::vector<float> &LPF1, const std::vector<float> &HPF1,
                  const std::vector<float> &BLURRED2, const std::vector<float> &LPF2,
                  const std::vector<float> &I_OPL, const std::vector<float> &V_BIP,
                  const std::vector<float> &Q_V_BIP, const std::vector<float> &V_BIP_LPF,
                  const std::vector<float> &HPF2, const std::vector<float> &I_Gang,
                  const std::vector<float> &V_LIF, const std::vector<unsigned char> &fired)
{

#ifdef SAVE_TO_FOLDER
    // Create output directories
    const std::vector<std::string> output_dirs = {
        "out", "out/imgs", "out/LPF1", "out/LPF2", "out/BLUR2", "out/HPF1",
        "out/I_OPL", "out/VBIP", "out/QVBIP", "out/VBIP_LPF", "out/HPF2",
        "out/IGang", "out/VLIF", "out/fired"};

    for (const auto &dir : output_dirs)
    {
        ensure_directory_exists(dir);
    }

    std::cout << "Saving results..." << std::endl;

    for (int t = 0; t < T; ++t)
    {
        try
        {
#ifdef SAVE_ALL
            // Save input images
            save_frame_generic(imgs, t, T, H, W, "out/imgs/" + std::to_string(t) + ".npy");

            // Save intermediate results (only in mode 0)
            if (mode == 0)
            {
                save_frame_generic(LPF1, t, T, H, W, "out/LPF1/" + std::to_string(t) + ".npy");
                save_frame_generic(HPF1, t, T, H, W, "out/HPF1/" + std::to_string(t) + ".npy");
                save_frame_generic(LPF2, t, T, H, W, "out/LPF2/" + std::to_string(t) + ".npy");
                save_frame_generic(BLURRED2, t, T, H, W, "out/BLUR2/" + std::to_string(t) + ".npy");
                save_frame_generic(I_OPL, t, T, H, W, "out/I_OPL/" + std::to_string(t) + ".npy");
                save_frame_generic(V_BIP, t, T, H, W, "out/VBIP/" + std::to_string(t) + ".npy");
                save_frame_generic(Q_V_BIP, t, T, H, W, "out/QVBIP/" + std::to_string(t) + ".npy");
                save_frame_generic(V_BIP_LPF, t, T, H, W, "out/VBIP_LPF/" + std::to_string(t) + ".npy");
                save_frame_generic(HPF2, t, T, H, W, "out/HPF2/" + std::to_string(t) + ".npy");
                save_frame_generic(I_Gang, t, T, H, W, "out/IGang/" + std::to_string(t) + ".npy");
                save_frame_generic(V_LIF, t, T, H, W, "out/VLIF/" + std::to_string(t) + ".npy");
            }

            // Save firing events
            save_frame_generic(fired, t, T, H, W, "out/fired/" + std::to_string(t) + ".npy");
#endif

            if ((t + 1) % 10 == 0 || t == T - 1)
            {
                std::cout << "Saved " << (t + 1) << "/" << T << " frames" << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error saving frame " << t << ": " << e.what() << std::endl;
        }
    }
#endif
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <image_directory> <mode> [options]\n"
                  << "  image_directory: Path to directory containing image sequence\n"
                  << "  mode: 0 (full output) or 1 (fired events only)\n"
                  << "Options:\n"
                  << "  --help, -h: Show this help message\n"
                  << std::endl;
        return 1;
    }

    fs::path cwd = fs::current_path();
    fs::path directory_path = argv[1];
    int mode = std::stoi(argv[2]);

    std::cout << "Hello, Silicon Retina Kernel!" << std::endl;
    std::cout << "Current working directory: " << cwd << std::endl;
    std::cout << "Input directory: " << directory_path << std::endl;
    std::cout << "Processing mode: " << mode << " (" << (mode == 0 ? "full output" : "fired events only") << ")" << std::endl;

    // Configuration
    SiliconRetinaConfig config;
    config.print();

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

    sort(files.begin(), files.end());

    int T = files.size();
    int W, H;
    if (T == 0)
    {
        std::cout << "No PNG files are found in the directory: " << directory_path << std::endl;
        return 1;
    }

    std::cout << "Found " << T << " PNG files in the directory: " << directory_path << std::endl;

    bool first_file = true;

    std::vector<float> imgs;

    int frame = 0;
    for (auto &el : files)
    {
        cv::Mat img8 = cv::imread(el.string(), cv::IMREAD_GRAYSCALE);
        if (first_file)
        {
            W = img8.cols;
            H = img8.rows;
            imgs = std::vector<float>(T * W * H, 0.0f);
            first_file = false;
        }

        // convert to float
        cv::Mat imgf;
        img8.convertTo(imgf, CV_32F, 1.0f / 255.0f);

        // copy directly into the flat buffer
        float *dst = imgs.data() + frame * H * W;

        // cv::Mat is row-major and continuuous, tthe big memcpy should work
        std::memcpy(dst, imgf.ptr<float>(), sizeof(float) * W * H);
        ++frame;
    }

    const size_t frame_size = static_cast<size_t>(H) * W;
    const size_t full_size = static_cast<size_t>(T) * frame_size;
    const size_t buffer_size = (mode == 0) ? full_size : 2 * frame_size;

    std::vector<float> LPF1(buffer_size, 0.0f);
    std::vector<float> HPF1(buffer_size, 0.0f);
    std::vector<float> BLURRED2(buffer_size, 0.0f);
    std::vector<float> LPF2(buffer_size, 0.0f);
    std::vector<float> I_OPL(buffer_size, 0.0f);
    std::vector<float> V_BIP(buffer_size, 0.0f);
    std::vector<float> Q_V_BIP(buffer_size, 0.0f);
    std::vector<float> V_BIP_LPF(buffer_size, 0.0f);
    std::vector<float> HPF2(buffer_size, 0.0f);
    std::vector<float> I_Gang(buffer_size, 0.0f);
    std::vector<float> V_LIF(buffer_size, 0.0f);
    std::vector<unsigned char> fired(full_size, 0);

    // Allocate device memory using RAII wrappers
    CudaMemory<float> d_imgs(full_size);
    CudaMemory<float> d_LPF1(buffer_size);
    CudaMemory<float> d_HPF1(buffer_size);
    CudaMemory<float> d_BLURRED2(buffer_size);
    CudaMemory<float> d_LPF2(buffer_size);
    CudaMemory<float> d_I_OPL(buffer_size);
    CudaMemory<float> d_V_BIP(buffer_size);
    CudaMemory<float> d_Q_V_BIP(buffer_size);
    CudaMemory<float> d_V_BIP_LPF(buffer_size);
    CudaMemory<float> d_HPF2(buffer_size);
    CudaMemory<float> d_I_Gang(buffer_size);
    CudaMemory<float> d_V_LIF(buffer_size);
    CudaMemory<unsigned char> d_fired(full_size);

    // Copy input data to device
    std::cout << "Copying data to GPU..." << std::endl;
    auto begin = std::chrono::high_resolution_clock::now();
    d_imgs.copyFromHost(imgs);
    d_LPF1.copyFromHost(LPF1);
    d_HPF1.copyFromHost(HPF1);
    d_BLURRED2.copyFromHost(BLURRED2);
    d_LPF2.copyFromHost(LPF2);
    d_I_OPL.copyFromHost(I_OPL);
    d_V_BIP.copyFromHost(V_BIP);
    d_Q_V_BIP.copyFromHost(Q_V_BIP);
    d_V_BIP_LPF.copyFromHost(V_BIP_LPF);
    d_HPF2.copyFromHost(HPF2);
    d_I_Gang.copyFromHost(I_Gang);
    d_V_LIF.copyFromHost(V_LIF);
    d_fired.copyFromHost(fired);

    // Run processing
    double duration_ms = run_silicon_retina_processing(
        config, mode, T, H, W,
        d_imgs, d_LPF1, d_HPF1, d_BLURRED2, d_LPF2, d_I_OPL,
        d_V_BIP, d_Q_V_BIP, d_V_BIP_LPF, d_HPF2, d_I_Gang, d_V_LIF, d_fired);

    if (mode == 0)
    {
        d_LPF1.copyToHost(LPF1);
        d_HPF1.copyToHost(HPF1);
        d_BLURRED2.copyToHost(BLURRED2);
        d_LPF2.copyToHost(LPF2);
        d_I_OPL.copyToHost(I_OPL);
        d_V_BIP.copyToHost(V_BIP);
        d_Q_V_BIP.copyToHost(Q_V_BIP);
        d_V_BIP_LPF.copyToHost(V_BIP_LPF);
        d_HPF2.copyToHost(HPF2);
        d_I_Gang.copyToHost(I_Gang);
        d_V_LIF.copyToHost(V_LIF);
    }
    d_fired.copyToHost(fired);
    auto end = std::chrono::high_resolution_clock::now();

    double duration_ms_total = std::chrono::duration<double, std::milli>(end - begin).count();
    std::cout << std::fixed << std::setprecision(3)
              << "Processing completed in " << duration_ms << " ms" << std::endl;
    std::cout << "Throughput: " << (T * 1000.0 / duration_ms) << " frames/second" << std::endl;
    std::cout << "Including copying results copied back in " << duration_ms_total << " ms" << std::endl;
    std::cout << "Throughput  " << (T * 1000.0 / duration_ms_total) << " frames/second" << std::endl;

    // Save results
    save_results(mode, T, H, W, imgs, LPF1, HPF1, BLURRED2, LPF2, I_OPL,
                 V_BIP, Q_V_BIP, V_BIP_LPF, HPF2, I_Gang, V_LIF, fired);

    std::cout << "Processing completed successfully!" << std::endl;

    return 0;
}