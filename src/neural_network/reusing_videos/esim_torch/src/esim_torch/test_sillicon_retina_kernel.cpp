#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "sillicon_retina_kernel.h"

#define SAVE_TO_FOLDER
#define SAVE_ALL


using namespace std;
namespace fs = std::filesystem;

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

void save_frame_as_png(const std::vector<float> &imgs,
                       int frame_idx,
                       int T, int H, int W,
                       const std::string &out_path)
{
    // sanity checks
    if (frame_idx < 0 || frame_idx >= T)
    {
        throw std::out_of_range("frame_idx out of bounds");
    }

    if ((int)imgs.size() != T * H * W)
    {
        throw std::runtime_error("imgs.size() doesn't match T*H*W");
    }

    const float *img_ptr = imgs.data() + frame_idx * H * W;

    cv::Mat img_f(H, W, CV_32F, (void *)img_ptr);

    // normalize / scale to 0-255
    cv::Mat img_u8;

    img_f.convertTo(img_u8, CV_8U, 255.0, 0.0);

    // save to disk
    if (!cv::imwrite(out_path, img_u8))
    {
        throw std::runtime_error("Failed to save image to " + out_path);
    }
}

void save_frame_as_png(const std::vector<char> &imgs,
                       int frame_idx,
                       int T, int H, int W,
                       const std::string &out_path)
{
    // sanity checks
    if (frame_idx < 0 || frame_idx >= T)
    {
        throw std::out_of_range("frame_idx out of bounds");
    }

    if ((int)imgs.size() != T * H * W)
    {
        throw std::runtime_error("imgs.size() doesn't match T*H*W");
    }

    const uint8_t *img_ptr = imgs.data() + frame_idx * H * W;
    cv::Mat img_u8(H, W, CV_8U, (void *)img_ptr);

    // save to disk
    if (!cv::imwrite(out_path, img_u8))
    {
        throw std::runtime_error("Failed to save image to " + out_path);
    }
}


int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <path to the folder directory>" << std::endl;
        return 1;
    }

    cout << "Hello, Silicon Retina Kernel!" << endl;

    fs::path directory_path = argv[1];

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

    std::vector<float> LPF1(T * H * W, 0.0f);
    std::vector<float> HPF1(T * H * W, 0.0f);
    std::vector<float> BLURRED2(T * H * W, 0.0f);
    std::vector<float> LPF2(T * H * W, 0.0f);
    std::vector<float> I_OPL(T * H * W, 0.0f);
    std::vector<float> V_BIP(T * H * W, 0.0f);
    std::vector<float> Q_V_BIP(T * H * W, 0.0f);
    std::vector<float> V_BIP_LPF(T * H * W, 0.0f);
    std::vector<float> HPF2(T * H * W, 0.0f);
    std::vector<float> I_Gang(T * H * W, 0.0f);
    std::vector<float> V_LIF(T * H * W, 0.0f);
    std::vector<char>  fired(T * H * W, 0);

    float *d_imgs;
    float *d_LPF1;
    float *d_HPF1;
    float *d_BLURRED2;
    float *d_LPF2;
    float *d_I_OPL;
    float *d_V_BIP;
    float *d_Q_V_BIP;
    float *d_V_BIP_LPF;
    float *d_HPF2;
    float *d_I_Gang;
    float *d_V_LIF;
    char  *d_fired;

    float alpha1 = 0.5f;
    float alpha2 = 0.5f;
    float V_BIP_LPF_alpha = 0.5f;
    float lambda_OPL = 0.5f;
    float G_A_zero = 0.5f;
    float lambda_A = 0.5f;
    float p1 = 0.5f;
    float I_bias = 0.5f;
    float p2 = 0.5f;
    float p3 = 0.5f;
    float U_rest = 0.5f;
    float U_th = 0.5f;
    float U_reset = 0.5f;

    cudaMalloc(&d_imgs, imgs.size() * sizeof(float));
    cudaMalloc(&d_LPF1, LPF1.size() * sizeof(float));
    cudaMalloc(&d_HPF1, HPF1.size() * sizeof(float));
    cudaMalloc(&d_BLURRED2, BLURRED2.size() * sizeof(float));
    cudaMalloc(&d_LPF2, LPF2.size() * sizeof(float));
    cudaMalloc(&d_I_OPL, I_OPL.size() * sizeof(float));
    cudaMalloc(&d_V_BIP, V_BIP.size() * sizeof(float));
    cudaMalloc(&d_Q_V_BIP, Q_V_BIP.size() * sizeof(float));
    cudaMalloc(&d_V_BIP_LPF, V_BIP_LPF.size() * sizeof(float));
    cudaMalloc(&d_HPF2, HPF2.size() * sizeof(float));
    cudaMalloc(&d_I_Gang, I_Gang.size() * sizeof(float));
    cudaMalloc(&d_V_LIF, V_LIF.size() * sizeof(float));
    cudaMalloc(&d_fired, fired.size() * sizeof(uint8_t));

    // Copy data to device
    cudaMemcpy(d_imgs, imgs.data(), imgs.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_LPF1, LPF1.data(), LPF1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_HPF1, HPF1.data(), HPF1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BLURRED2, BLURRED2.data(), BLURRED2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_LPF2, LPF2.data(), LPF2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_I_OPL, I_OPL.data(), I_OPL.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_BIP, V_BIP.data(), V_BIP.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q_V_BIP, Q_V_BIP.data(), Q_V_BIP.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_BIP_LPF, V_BIP_LPF.data(), V_BIP_LPF.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_HPF2, HPF2.data(), HPF2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_I_Gang, I_Gang.data(), I_Gang.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_LIF, V_LIF.data(), V_LIF.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fired, fired.data(), fired.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);  

    run_silicon_retina_model(
        d_imgs,
        d_LPF1,
        d_HPF1,
        d_BLURRED2,
        d_LPF2,
        d_I_OPL,
        d_V_BIP,
        d_Q_V_BIP,
        d_V_BIP_LPF,
        d_HPF2,
        d_I_Gang,
        d_V_LIF,
        d_fired,
        T,
        H,
        W,
        alpha1,
        alpha2,
        V_BIP_LPF_alpha,
        lambda_OPL,
        G_A_zero,
        lambda_A,
        p1,
        I_bias,
        p2,
        p3,
        U_rest,
        U_th,
        U_reset);

    cudaDeviceSynchronize();

    // 5. Copy results back
    cudaMemcpy(LPF1.data(), d_LPF1, LPF1.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(HPF1.data(), d_HPF1, HPF1.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(BLURRED2.data(), d_BLURRED2, BLURRED2.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(LPF2.data(), d_LPF2, LPF2.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(I_OPL.data(), d_I_OPL, I_OPL.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(V_BIP.data(), d_V_BIP, V_BIP.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Q_V_BIP.data(), d_Q_V_BIP, Q_V_BIP.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(V_BIP_LPF.data(), d_V_BIP_LPF, V_BIP_LPF.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(HPF2.data(), d_HPF2, HPF2.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(I_Gang.data(), d_I_Gang, I_Gang.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(V_LIF.data(), d_V_LIF, V_LIF.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fired.data(), d_fired, fired.size() * sizeof(int), cudaMemcpyDeviceToHost);  

    // 6. Print a few results

    for (int t = 0; t < T; ++t)
    {
#ifdef SAVE_TO_FOLDER
#ifdef SAVE_ALL
        std::string out_path = "out/imgs/" + std::to_string(t) + ".png";
        save_frame_as_png(imgs, t, T, H, W, out_path);

        out_path = "out/LPF1/" + std::to_string(t) + ".png";
        save_frame_as_png(LPF1, t, T, H, W, out_path);

        out_path = "out/HPF1/" + std::to_string(t) + ".png";
        save_frame_as_png(HPF1, t, T, H, W, out_path);

        out_path = "out/LPF2/" + std::to_string(t) + ".png";
        save_frame_as_png(LPF2, t, T, H, W, out_path);

        out_path = "out/BLUR2/" + std::to_string(t) + ".png";
        save_frame_as_png(BLURRED2, t, T, H, W, out_path);

        out_path = "out/IOPL/" + std::to_string(t) + ".png";
        save_frame_as_png(I_OPL, t, T, H, W, out_path);

        out_path = "out/VBIP/" + std::to_string(t) + ".png";
        save_frame_as_png(V_BIP, t, T, H, W, out_path);

        out_path = "out/QVBIP/" + std::to_string(t) + ".png";
        save_frame_as_png(Q_V_BIP, t, T, H, W, out_path);

        out_path = "out/VBIP_LPF/" + std::to_string(t) + ".png";
        save_frame_as_png(V_BIP_LPF, t, T, H, W, out_path);
        
        out_path = "out/HPF2/" + std::to_string(t) + ".png";
        save_frame_as_png(HPF2, t, T, H, W, out_path);

        out_path = "out/IGang/" + std::to_string(t) + ".png";
        save_frame_as_png(I_Gang, t, T, H, W, out_path);
        
        out_path = "out/VLIF/" + std::to_string(t) + ".png";
        save_frame_as_png(V_LIF, t, T, H, W, out_path);

        out_path = "out/fired/" + std::to_string(t) + ".png";
        save_frame_as_png(fired, t, T, H, W, out_path);

#else
        std::string out_path = "out/imgs_lpf1_" + std::to_string(t) + ".png";
        save_frame_as_png(HPF1, t, T, H, W, out_path);

#endif

#else

        std::cout << "\n\n frame " << t << ": ";
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                std::cout << imgs[t * H * W + y * W + x] << " ";
            }
            std::cout << std::endl;
        }
#endif
    }

    // 7. Free
    cudaFree(d_imgs);
    cudaFree(d_LPF1);
    cudaFree(d_HPF1);
    cudaFree(d_BLURRED2);
    cudaFree(d_LPF2);
    cudaFree(d_I_OPL);
    cudaFree(d_V_BIP);
    cudaFree(d_Q_V_BIP);
    cudaFree(d_V_BIP_LPF);
    cudaFree(d_HPF2);
    cudaFree(d_I_Gang);
    cudaFree(d_V_LIF);
    cudaFree(d_fired); 

    return 0;
}