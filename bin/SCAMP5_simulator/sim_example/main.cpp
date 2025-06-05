/*************************************************************************
 * SCAMP Vision Chip Development System Library
 *------------------------------------------------------------------------
 * Copyright (c) 2020 The University of Manchester. All Rights Reserved.
 *
 *************************************************************************/
 /*
  * Scamp5d M0 Example 1 - Image Capturing and Host Display
  *
  */

  // 256 x 128

#define VISUALIZE_DEBUG
#define SIMULATION _MSC_VER
#include <iostream>
#include <scamp5.hpp>
#include <scamp5_algorithm.hpp>
#include <direct.h>
#include <sstream>
#define getcwd _getcwd // Use _getcwd on Windows
#include <limits.h>
#include <iomanip>
#include <string>
#include <math.h>
#include <format>
#define _CRT_SECURE_NO_WARNINGS


#define DREG_IS_PE					S0
#define DREG_IS_STORAGE				S1
#define DREG_BUFF					S2

#define GAUSS1_ITER					1
#define GAUSS2_VERT_ITER			2
#define GAUSS2_HOR_ITER				1

#define PE_LPF_PREV					B
#define PE_DIVA_IMM1				E
#define PE_DIVA_IMM2				F

#define STO_LPF1					B
#define STO_LPF2					C
#define STO_LPF3					D
#define STO_LPF4					E


// bunch of global variables
bool first_run = true;
int MSB;
int G0 = 0;
int Ibias = 0;
int threshold = 0;
int V_rest = 0;
int U_reset = 30;
int run_blur_1 = 0;
int run_avg_1 = 0;
scamp5_algorithm::image_loader *img_loader;


// B^{t+1} := B + \alpha * (A - B)
struct LPF_fixed
{
	int no_of_bits_for_alpha = 4;
	int alpha = 0b0010;
};

struct fixed_point
{
	int no_of_bits = 4;
	int param = 0b1000;
};

using namespace SCAMP5_PE;

enum save_type
{
	INPUT_IMG, 
	LPF1,
	HPF1,
	HPF1_BLUR,
	LPF2,
	OPL,
	VBIP,
	VBIP_SUB_OPL,
	F1,
	F2,
	LPF4, 
	V_MEM,
	EVENTS, 
	CWD
};

void save_img(save_type mode, areg_t reg, int frame_count)
{
	char buffer[50];

	switch (mode)
	{
	case INPUT_IMG:
		sprintf_s(buffer, sizeof(buffer), "output\\input\\%d.bmp", frame_count);
		break;
	case LPF1:
		sprintf_s(buffer, sizeof(buffer), "output\\lpf1\\%d.bmp", frame_count);
		break;
	case HPF1:
		sprintf_s(buffer, sizeof(buffer), "output\\hpf1\\%d.bmp", frame_count);
		break;
	case HPF1_BLUR:
		sprintf_s(buffer, sizeof(buffer), "output\\hpf1_blur\\%d.bmp", frame_count);
		break;
	case LPF2:
		sprintf_s(buffer, sizeof(buffer), "output\\lpf2\\%d.bmp", frame_count);
		break;
	case OPL:
		sprintf_s(buffer, sizeof(buffer), "output\\opl\\%d.bmp", frame_count);
		break;
	case VBIP:
		sprintf_s(buffer, sizeof(buffer), "output\\vbip\\%d.bmp", frame_count);\
		break;
	case VBIP_SUB_OPL:
		sprintf_s(buffer, sizeof(buffer), "output\\vbip_sub_opl\\%d.bmp", frame_count); \
			break;
	case F1:
		sprintf_s(buffer, sizeof(buffer), "output\\f1\\%d.bmp", frame_count);
		break;
	case F2:
		sprintf_s(buffer, sizeof(buffer), "output\\f2\\%d.bmp", frame_count);
		break;
	case LPF4:
		sprintf_s(buffer, sizeof(buffer), "output\\lpf4\\%d.bmp", frame_count);
		break;
	case V_MEM:
		sprintf_s(buffer, sizeof(buffer), "output\\vmem\\%d.bmp", frame_count);
		break;
	case EVENTS:
		sprintf_s(buffer, sizeof(buffer), "output\\events\\%d.bmp", frame_count);
		break;
	default:
		break;
	}

	vs_sim::save_image(reg, buffer);
}

void save_img(save_type mode, dreg_t reg, int frame_count)
{
	char buffer[50];

	switch (mode)
	{;
	case EVENTS:
		sprintf_s(buffer, sizeof(buffer), "output\\events\\%d.bmp", frame_count);
		break;
	case CWD:
		sprintf_s(buffer, sizeof(buffer), "HERE.bmp");
	default:
		break;
	}

	vs_sim::save_image(reg, buffer);
}

void save_blured_image_to_A(int frame_count)
{
	// load the image
	std::ostringstream oss;

	// Format the string: pad the integer to 4 digits with leading zeros
	oss << "E:\\PROJS\\sillicon retina\\prototyping\\dataset\\" << std::setw(4) << std::setfill('0') << frame_count << ".bmp";
	img_loader->load_areg_image(oss.str().c_str());



	// apply blur to the image
	if (run_blur_1)
	{
		scamp5_kernel_begin();
		SET(R1, R2);
		gauss(A, A, GAUSS1_ITER);
		//mov(C, F);
		scamp5_kernel_end();
	}

	// checkpoint : blurred or nor image in A
	save_img(save_type::INPUT_IMG, A, frame_count);
}

void compute_1st_lpf_and_store_it_to_B(LPF_fixed &lpf, int frame_count)
{
	// load the LPF 1 value from storage 
	scamp5_kernel_begin();
		WHERE(DREG_IS_STORAGE);
			bus(NEWS, STO_LPF1);

		WHERE(DREG_IS_PE);
			bus(PE_LPF_PREV, XS);
	scamp5_kernel_end();

	// STORE the A - B in C
	scamp5_kernel_begin();
		sub(C, A, PE_LPF_PREV);
		diva(C, PE_DIVA_IMM1, PE_DIVA_IMM2);
	scamp5_kernel_end();

	// compute the LPF 
	int bits = 1 << (lpf.no_of_bits_for_alpha - 1);
	while (bits)
	{
		if (lpf.alpha & bits)
		{
			scamp5_kernel_begin();
			add(PE_LPF_PREV, PE_LPF_PREV, C);
			scamp5_kernel_end();
		}
		bits >>= 1;
		scamp5_kernel_begin();
		diva(C, PE_DIVA_IMM1, PE_DIVA_IMM2);
		scamp5_kernel_end();
	}

	// now store back the new LPF1 back to the storage 
	scamp5_kernel_begin();
		// 
		WHERE(DREG_IS_PE);
			bus(NEWS, PE_LPF_PREV);

		WHERE(DREG_IS_STORAGE);
			bus(STO_LPF1, XN);
	scamp5_kernel_end();

	// checkpoint : the LPF1 is now in B 
	save_img(save_type::LPF1, B, frame_count);
}

void compute_1st_hpf_and_store_it_in_A_and_C(int frame_count)
{
	// compute the High Pass Filter 1, and store it in C and A
	scamp5_kernel_begin();
	WHERE(DREG_IS_PE);
		sub(C, A, B);
		mov(A, C);
	scamp5_kernel_end();

	// checkpoint : the HPF1 is now in A and C
	save_img(save_type::HPF1, A, frame_count);
}

void compute_blur_of_HPF_and_store_it_in_A(int frame_count)
{
	scamp5_kernel_begin();
		// 1. move the the PE.HPF (A) to STO.A
		WHERE(DREG_IS_PE);
		bus(NEWS, A);

		WHERE(DREG_IS_STORAGE);
		bus(A, XN);

		ALL();
		// 2. now apply the vertical blur 2 twice, nad horizontal blur once
		SET(R1);
		CLR(R2);
		gaussv(A, A, GAUSS2_VERT_ITER);

		CLR(R1);
		SET(R2);
		gaussh(A, A, GAUSS2_HOR_ITER);
	scamp5_kernel_end();

	// checkpoint : the blurred HPF1 is now in A 
	save_img(save_type::HPF1_BLUR, A, frame_count);
}

void compute_2nd_LPF_of_blurred_HPF_and_store_it_to_B(LPF_fixed &lpf, int frame_count)
{
	// compute the 2nd LPF 
	// 1. LOAD the previous LPF2 map from storage
	scamp5_kernel_begin();
		WHERE(DREG_IS_STORAGE);
			bus(NEWS, STO_LPF2);

		WHERE(DREG_IS_PE);
			bus(B, XS);
	scamp5_kernel_end();


	// 2. compute the difference between the incoming map (A) and previous LPF (B)
	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			sub(A, A, B);
			diva(A, PE_DIVA_IMM1, PE_DIVA_IMM2);
	scamp5_kernel_end();

	// 3. actually compute the LPF2, where B is storing the previous LPF map and A is incoming blurred map
	int value = 1 << (lpf.no_of_bits_for_alpha - 1);
	while (value)
	{

		if (value & lpf.alpha)
		{
			scamp5_kernel_begin();
				add(B, B, A);
			scamp5_kernel_end();
		}

		scamp5_kernel_begin();
			diva(A, PE_DIVA_IMM1, PE_DIVA_IMM2);
		scamp5_kernel_end();
		value >>= 1;
	}

	// 4. store back the result from PE to STO
	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			bus(NEWS, B);

		WHERE(DREG_IS_STORAGE);
			bus(STO_LPF2, XN);
	scamp5_kernel_end();

	// checkpoint : the LPF2 is in B 
	save_img(save_type::LPF2, B, frame_count);
}

void compute_OPL_and_store_it_B(int frame_count)
{
	// Compute the difference between the computed HPF and blurred, low pass filter 
	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
		sub(A, B, C);
	scamp5_kernel_end();

	// checkpoint : the LPF2 is in B 
	save_img(save_type::OPL, B, frame_count);
}

void compute_the_Vbip_and_store_it_to_B(int frame_count, LPF_fixed &lpf)
{
	// 1. load the lpf from storage 
	scamp5_kernel_begin();
		WHERE(DREG_IS_STORAGE);
			bus(NEWS, STO_LPF3);

		WHERE(DREG_IS_PE);
			bus(B, XS);
	scamp5_kernel_end();

	// 2. compute the difference between the incoming (A) and the LPF (B)
	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			sub(C, A, B);
			diva(C, PE_DIVA_IMM1, PE_DIVA_IMM2);
	scamp5_kernel_end();

	// TODO : this needs to be adaptive, so coefficients for alpha has to be stored in DREG
	int value = 1 << (lpf.no_of_bits_for_alpha - 1);
	while (value)
	{
		if (value & lpf.alpha)
		{
			scamp5_kernel_begin();
				add(B, B, C);
			scamp5_kernel_end();
		}
		scamp5_kernel_begin();
			diva(C, PE_DIVA_IMM1, PE_DIVA_IMM2);
		scamp5_kernel_end();
		value >>= 1;
	}

	// 3. save the new LPF from PE to storage
	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			bus(NEWS, B);

		WHERE(DREG_IS_STORAGE);
			bus(STO_LPF3, XN);
	scamp5_kernel_end();

	// the value in B corresponds now to V_{Bip}
	save_img(save_type::VBIP, B, frame_count);
}

void compute_the_diff_of_VBIP_and_OPL_store_it_A(int frame_count)
{
	// to compute now the value that will be converted to Ganglion Cell current
	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			sub(A, A, B);
	scamp5_kernel_end();
	
	save_img(save_type::VBIP_SUB_OPL, A, frame_count);
}

void compute_f1_on_VBip_and_store_it_in_C(int frame_count, fixed_point &lambda)
{
	// compute now the f_1, which is G_0 + \lambda * V_{BIP}^{2} 
	// which here is approximated by G_0 + \lambda * |V_{BIP}|

	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			abs(C, B);			// FLAG will be set to ALL() !

		WHERE(DREG_IS_PE);
			diva(C, PE_DIVA_IMM1, PE_DIVA_IMM2);
	scamp5_kernel_end();

	scamp5_load_in(F, 0);

	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			mov(B, F);
	scamp5_kernel_end();

	int el = 1 << (lambda.no_of_bits - 1);

	while (el)
	{
		if (el & lambda.param)
		{
			scamp5_kernel_begin();
				add(B, B, C);
			scamp5_kernel_end();
		}
		scamp5_kernel_begin();
			diva(C, PE_DIVA_IMM1, PE_DIVA_IMM2);
		scamp5_kernel_end();
		el >>= 1;
	}

	// at this point a \lambda (which is a scaling constant less then 1.0) is multiplied by |V_{BIP}| approximately
	// B stores now the \lambda |V_{BIP}|
	// to finish the computation, we do need to load the G0 
	scamp5_load_in(F, G0);
	scamp5_kernel_begin()
		mov(C, F);
	scamp5_kernel_end();

	scamp5_kernel_begin();
		add(C, C, B);
	scamp5_kernel_end();

	save_img(save_type::F1, C, frame_count);
}

void compute_4th_LPF_of_f1_and_store_B(int frame_count, LPF_fixed &lpf)
{
	// compute now the LPF4 
	// 1. load the previous LPF from storage value into PE
	scamp5_kernel_begin();
		WHERE(DREG_IS_STORAGE);
			bus(NEWS, STO_LPF4);

		WHERE(DREG_IS_PE);
			bus(B, XS);
	scamp5_kernel_end();

	// 2. compute now the the difference of incoming (C) and previous (B)
	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			sub(C, C, B);
			diva(C, PE_DIVA_IMM1, PE_DIVA_IMM2);
	scamp5_kernel_end();


	// compute the LPF4
	// 3.
	int val = 1 << (lpf.no_of_bits_for_alpha - 1);
	while (val)
	{
		if (val & lpf.alpha)
		{
			scamp5_kernel_begin();
			add(B, B, C);
			scamp5_kernel_end();
		}

		scamp5_kernel_begin();
		diva(C, PE_DIVA_IMM1, PE_DIVA_IMM2);
		scamp5_kernel_end();
		val >>= 1;
	}

	// 4. now store the result back to storage
	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			bus(NEWS, B);

		WHERE(DREG_IS_STORAGE);
			bus(STO_LPF4, XN);
	scamp5_kernel_end();

	save_img(save_type::LPF4, B, frame_count);
}

void update_the_feedback_based_on_B(int frame_count)
{

}

void compute_f2_store_A(int frame_count)
{
	// compute the max(0, A) + Ibias

	// zero out Fs 
	scamp5_load_in(F, 0);
	
	// now compute the f_2, based on A 
	scamp5_kernel_begin();
		where(A);
		AND(DREG_BUFF, DREG_IS_PE, FLAG);

		WHERE(DREG_BUFF);
			mov(A, F);

		ALL();
	scamp5_kernel_end();

	// add a bias to the current
	scamp5_load_in(F, Ibias);
	
	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			add(A, A, F);
	scamp5_kernel_end();

	save_img(save_type::F2, A,  frame_count);
}


/**
* 
*		The computation here looks like following 
*		the input is I_{gang}
* 
*		du/dt = 1/tau ( R * I_{gang}(t)  - |u(t) - u_{rest}| )
* 
*		and is being solved by euler method 
* 
*		u(t + \delta t) = u(t) + \delta t / tau (R * I_{gang}(t) - |u(t) - u_{rest}|)
* 
*		where the computation will be done as following 
* 
*		u(t) + p_{2} * (p_{1} * I_{gang}(t) - |u(t) - u_{rest}|)
*
*		where p_{2} and p_{1} will represent the 5 bit fixed point values in a for b_1.b_2 b_3 b_4 b_5
* 
*/
void update_membranes_on_LIFs_generate_events(int frame_count, fixed_point &p1, fixed_point &p2)
{

	// firstly compute the p_{1} * I_{gang}(t), where I_{gang} is stored in A
	// this will be done 

	// 1. zero out B
	scamp5_load_in(F, 0);
	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			mov(B, F);
	scamp5_kernel_end();

	// 2. multiply by param p1
	int val = 1 << (p1.no_of_bits - 1);
	while (val)
	{
		if (val & p1.param)
		{
			scamp5_kernel_begin();
				add(B, B, A);
			scamp5_kernel_end();
		}
	
		scamp5_kernel_begin();
			diva(A, PE_DIVA_IMM1, PE_DIVA_IMM2);
		scamp5_kernel_end();
		
		val >>= 1;
	}

	scamp5_load_in(F, V_rest);

	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			// compute the |U(t) - U_{rest}(t)|
			sub(A, D, F);
			abs(C, A);
			mov(A, C);

			// now compute p_1 I_{gang}(t) - |U(t) - U_{rest}(t)|
			sub(B, B, A);

	scamp5_kernel_end();

	scamp5_load_in(F, 0);

	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			mov(A, F);
	scamp5_kernel_end();

	val = 1 << (p2.no_of_bits - 1);
	while (val)
	{
		if (val & p2.param)
		{
			scamp5_kernel_begin();
				add(A, A, B);
			scamp5_kernel_end();
		}

		val >>= 1;
		scamp5_kernel_begin();
			diva(B, PE_DIVA_IMM1, PE_DIVA_IMM2);
		scamp5_kernel_end();
	}
	scamp5_kernel_begin();
		add(D, D, A);
	scamp5_kernel_end();

	scamp5_load_in(F, U_reset);
	scamp5_load_in(A, threshold);

	scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
			sub(B, D, A);

		where(B);
			MOV(DREG_BUFF, FLAG);
			AND(DREG_BUFF, DREG_BUFF, DREG_IS_PE);

		WHERE(DREG_BUFF);
			sub(D, D, F);

	scamp5_kernel_end();

	save_img(save_type::V_MEM, D, frame_count);
	save_img(save_type::V_MEM, DREG_BUFF, frame_count);
}


int main()
{
	 img_loader = new scamp5_algorithm::image_loader(A, R9, F, R10);


#ifdef SIMULATION
	vs_sim::config("server_ip", "127.0.0.1");
	vs_sim::config("server_port", "27715");
#endif

	vs_init();

#ifdef SIMULATION
	vs_sim::enable_keyboard_control();// this allow a few shortcuts to be used. e.g. 'Q' to quit.
	vs_sim::reset_model(3);// reset model is also used to configure the error model
#endif

	vs_gui_set_info(VS_M0_PROJECT_INFO_STRING);

	vs_gui_add_switch("run_blur_1", 0, &run_blur_1);
	vs_gui_add_switch("run_avg_1", 0, &run_avg_1);


	LPF_fixed lpf1 = { 4,			// width of alpha
					   0b0100		// alpha value
	};

	LPF_fixed lpf2 = { 4,
					   0b0011,
	};

	LPF_fixed lpf3 = { 4,
					   0b0010,
	};

	LPF_fixed lpf4 = { 4,
					  0b0010
	};

	LPF_fixed lpf5 = { 4,
					  0b0001
	};

	fixed_point p1 = {
		5,
		0b01000
	};

	fixed_point p2 = {
		5,
		0b01000
	};

	fixed_point lambda = {
		5,
		0b10000
	};


	vs_gui_add_slider("alpha1_value", 0, 255, lpf1.alpha, &lpf1.alpha);
	vs_gui_add_slider("alpha1_bits", 1, 6, 4, &lpf1.no_of_bits_for_alpha);

	vs_gui_add_slider("alpha2_value", 0, 255, lpf2.alpha, &lpf2.alpha);
	vs_gui_add_slider("alpha2_bits", 1, 6, 4, &lpf2.no_of_bits_for_alpha);

	vs_gui_add_slider("alpha3_value", 0, 255, lpf3.alpha, &lpf3.alpha);
	vs_gui_add_slider("alpha3_bits", 1, 6, 4, &lpf3.no_of_bits_for_alpha);

	vs_gui_add_slider("alpha4_value", 0, 255, lpf4.alpha, &lpf4.alpha);
	vs_gui_add_slider("alpha4_bits", 1, 6, 4, &lpf4.no_of_bits_for_alpha);

	vs_gui_add_slider("alpha5_value", 0, 255, lpf5.alpha, &lpf5.alpha);
	vs_gui_add_slider("alpha5_bits", 1, 6, 4, &lpf5.no_of_bits_for_alpha);

	vs_gui_add_slider("G0", 0, 127, G0, &G0);
	vs_gui_add_slider("IBias", -128, 127, Ibias, &Ibias);
	vs_gui_add_slider("threshold", -128, 127, threshold, &threshold);
	vs_gui_add_slider("U_{rest}", -128, 127, V_rest, &V_rest);
	vs_gui_add_slider("U_{reset}", 0, 127, U_reset, &U_reset);

	vs_gui_add_slider("lambda", 0, 127, lambda.param, &lambda.param);
	vs_gui_add_slider("lambda_bits", 2, 5, lambda.no_of_bits, &lambda.no_of_bits);


	vs_gui_add_slider("V_{rest}", -128, 127, V_rest, &V_rest);

	char cwd[256];
	if (getcwd(cwd, sizeof(cwd)) != nullptr)
	{
		std::cout << "Current working directory : " << cwd << std::endl;
	}


#ifdef VISUALIZE_DEBUG
	auto display_1 = vs_gui_add_display("image", 0, 0, 1);
	auto display_2 = vs_gui_add_display("LPF", 0, 1, 1);
	auto display_3 = vs_gui_add_display("HPF", 0, 2, 1);
	auto display_4 = vs_gui_add_display("membranes", 1, 0, 1);
	auto display_5 = vs_gui_add_display("events", 1, 1, 1);
#endif


	scamp5_kernel_begin();
	ALL();
	scamp5_kernel_end();
	scamp5_load_in(A, 0);
	scamp5_load_in(B, 0);
	scamp5_load_in(C, 0);
	scamp5_load_in(D, 0);
	scamp5_load_in(E, 0);
	scamp5_load_in(F, 0);

	int frame_count = 0;
	// Frame Loop


	while (1) {
		vs_frame_loop_control();

		// load pattern
		scamp5_load_pattern(DREG_IS_STORAGE, 0b0000001, 0b00000000, 0b11111110, 0b11111111);
		scamp5_kernel_begin();
			NOT(DREG_IS_PE, DREG_IS_STORAGE);
		scamp5_kernel_end();
		save_img(save_type::CWD, DREG_IS_PE, 0);

		// loads the image and saves to A
		// blurs iit if requested
		save_blured_image_to_A(frame_count);

		// the computed value is now in A. Compute a low pass filter in B by
		// taking it from STO.B and compute with incoming A
		compute_1st_lpf_and_store_it_to_B(lpf1, frame_count);

		// compute the first HPF
		//compute_1st_hpf_and_store_it_in_A_and_C(frame_count);

		// compute the blur of the HPF
		//compute_blur_of_HPF_and_store_it_in_A(frame_count);

		// compute the 2nd LPF : the incoming map is in A, the previous map is loaded from storage
		// and saved into B
		//compute_2nd_LPF_of_blurred_HPF_and_store_it_to_B(lpf2, frame_count);

		// compyte the OPL by subtracting the previously captured HPF 1 (stored in C)
		// 
		//compute_OPL_and_store_it_B(frame_count);

		// Compute the 3rd low pass map
		//compute_the_Vbip_and_store_it_to_B(frame_count, lpf3);

		//compute_the_diff_of_VBIP_and_OPL_store_it_A(frame_count);

		//compute_f1_on_VBip_and_store_it_in_C(frame_count, lambda);

		
		// C now contains the \lambda |V_{BIP}| + G_0
		// and is incoming map to the LPF 4
		//compute_4th_LPF_of_f1_and_store_B(frame_count, lpf4);
		

		// using the value computed by 4th LPF 
		// update the feedback
		//update_the_feedback_based_on_B(frame_count);

		// compute the f2 from value in A and store it in A
		//compute_f2_store_A(frame_count);


		//update_membranes_on_LIFs_generate_events(frame_count, p1, p2);
		// events now stored in DREG_BUFF, membrane in D

		first_run = false;
		frame_count = (frame_count + 1) % 300;

#ifdef VISUALIZE_DEBUG
		if (vs_gui_is_on())
		{
			scamp5_output_image(A, display_1);
			//scamp5_output_image(B, display_2);
			//scamp5_output_image(C, display_3);
			//scamp5_output_image(E, display_4);
			//scamp5_output_image(R6, display_5);
		}
#endif

	}
	return 0;
}
