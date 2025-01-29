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

/*
#include <scamp5.hpp>
#define LOW_PASS_FILTER1_HIST_AREG B
#define LOW_PASS_FILTER2_HIST_AREG C
#define VISUALIZE
#define SIMULATION

#define LPF1_CNT_THRESH 10
#define THRESHOLD 10

#define ALPHA_LPF1_WIDTH  		   4
#define ALPHA_LPF2_WIDTH  		   3

const bool add_reminder = true;
const uint8_t initial_alpha_LPF1 = 0b0011;
dreg_t LPF1_BITS[ALPHA_LPF1_WIDTH] = { SCAMP5_PE::R3, SCAMP5_PE::R4, SCAMP5_PE::R5, SCAMP5_PE::R6 };


const uint8_t initial_alpha_LPF2 = 0b001;
dreg_t LPF2_BITS[ALPHA_LPF2_WIDTH] = { SCAMP5_PE::R7, SCAMP5_PE::R8, SCAMP5_PE::R9 };


using namespace SCAMP5_PE;

int LPF1_cnt = 0;
int LPF2_cnt = 0;
int low_pass_blur_iters1 = 1;
int low_pass_blur_iters2 = 1;

*/

/*
 * 	Analog Storage
 * 	A - low pass filter 1 past map
 * 	B - low pass filter 2 past map
 * 	C - processed image, and subtraction
 *
 *
 *                                   (alpha stored as analog)
 *   Image (C) -> Blur(C) -> Low pass of Blurred Image (C) to (A) -> High pass (store into A) ----------|
 *                                                                       |                              |
 *                                                                       |                              |
 *                                                                    Blur of high                      |
 *                                                                    pass (store into B)               |
 *                                                                       |                              |
 *                                                                       |                           -------
 *                                                                     Compute Low                   |     |
 *                                                                     Pass of B   --------------->  |  -  |
 *                                                                     into B                        |     |
 *                                                                     (Cheaper)                     |-----|
 */


/*
 // temporary : F, res -> C
void get_image_store_it_in_C()
{
	scamp5_kernel_begin();
	get_image(C, F);
	scamp5_kernel_end();
}

// temporary : F, res -> C
void blur_image_and_store_it_in_C()
{
	scamp5_dynamic_kernel_begin();
	SET(R1, R2);
	gauss(F, C, low_pass_blur_iters1);
	mov(C, F);
	scamp5_dynamic_kernel_end();
}

void load_alpha_into_registers(int alpha_to_load, int alpha_width, dreg_t* dreg_pointer)
{
	for (int i = 0; i < alpha_width; ++i)
	{
		bool bit = ((alpha_to_load >> (alpha_width - i - 1)) & 1);

		if (bit)
		{
			scamp5_dynamic_kernel_begin();
			SET(dreg_pointer[i]);
			scamp5_dynamic_kernel_end();
		}
		else
		{
			scamp5_dynamic_kernel_begin();
			CLR(dreg_pointer[i]);
			scamp5_dynamic_kernel_end();
		}
	}
}

// Intermediate F -> result in B
// applies the following formula to image B
// A := A + \alpha \cdot (C - A)
void apply_LPF_to_A_from_image_C(bool add_reminder = true)
{

	if (LPF1_cnt < LPF1_CNT_THRESH)
	{
		scamp5_kernel_begin();
		mov(A, C);
		scamp5_kernel_end();
		load_alpha_into_registers(initial_alpha_LPF1, ALPHA_LPF1_WIDTH, LPF1_BITS);
		LPF1_cnt += 1;
	}

	// store the C - A in D
	scamp5_kernel_begin();
	sub(D, C, A);
	mov(B, D);
	scamp5_kernel_end();

	for (int i = 0; i < ALPHA_LPF1_WIDTH; ++i)
	{
		scamp5_dynamic_kernel_begin();
		diva(D, E, F);
		WHERE(LPF1_BITS[i]);
		add(A, A, D);
		ALL();
		scamp5_dynamic_kernel_end();
	}

*/
	/*
	scamp5_kernel_begin();
		diva(D, E, F);
		add(A, A, D);
	scamp5_kernel_end();
	*/
/*
}

// temporary : , res -> C
void apply_HPF_to_C_from_C_and_LPF_from_A()
{
	scamp5_kernel_begin();
	sub(C, C, A);
	scamp5_kernel_end();
	*/
	/*
	scamp5_in(F, THRESHOLD);

	scamp5_kernel_begin();
		mov(E, F);
		sub(E, E, F);
		where(E);
		NOT(R0, FLAG);
		add(F, F, F);
		add(E, E, F);
		where(E);
	scamp5_kernel_end();

	scamp5_in(F, 0);
	scamp5_kernel_begin();
		mov(E, F);
		ALL();
		mov(C, E);
	scamp5_kernel_end();
	*/
/*
}

// temporary : F
void blur_image_2_and_store_it_in_D()
{
	scamp5_dynamic_kernel_begin();
	SET(R1, R2);
	gauss(F, C, low_pass_blur_iters2);
	mov(D, F);
	scamp5_dynamic_kernel_end();
}


void apply_LPF_to_B_from_image_D(bool add_reminder = true)
{

	if (LPF2_cnt < LPF1_CNT_THRESH)
	{
		scamp5_kernel_begin();
		mov(B, D);
		scamp5_kernel_end();
		LPF2_cnt += 1;
		load_alpha_into_registers(initial_alpha_LPF2, ALPHA_LPF2_WIDTH, LPF2_BITS);
	}

	// store the B - D in D
	scamp5_kernel_begin();
	sub(D, B, D);
	scamp5_kernel_end();

	for (int i = 0; i < ALPHA_LPF1_WIDTH; ++i)
	{
		scamp5_dynamic_kernel_begin();
		diva(D, E, F);
		WHERE(LPF2_BITS[i]);
		add(B, B, D);
		ALL();
		scamp5_dynamic_kernel_end();
	}
}

void apply_subtraction()
{
	scamp5_kernel_begin();
	sub(F, B, C);
	scamp5_kernel_end();
}

int main() {

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

#ifdef VISUALIZE
	auto display_1 = vs_gui_add_display("HPF", 0, 0, 1);
	auto display_2 = vs_gui_add_display("LPF", 0, 1, 1);
	auto display_3 = vs_gui_add_display("LPF2", 0, 2, 1);
	auto display_4 = vs_gui_add_display("diff", 1, 0, 1);
#endif

	int run_blur_1 = 1;
	vs_gui_add_switch("run_blur_1", 1, &run_blur_1);

	int run_LPF1 = 1;
	vs_gui_add_switch("run_LPF1", 1, &run_LPF1);

	int run_HPF = 1;
	vs_gui_add_switch("run_HPF", 1, &run_HPF);

	int run_blur_2 = 1;
	vs_gui_add_switch("run_blur_1", 1, &run_blur_2);

	int run_LPF2 = 1;
	vs_gui_add_switch("run_LPF2", 1, &run_LPF2);

	int run_subtraction = 1;
	vs_gui_add_switch("run_subtraction", 1, &run_subtraction);

	vs_gui_add_slider("Gauss iters 1", 1, 11, low_pass_blur_iters1, &low_pass_blur_iters1);
	vs_gui_add_slider("Gauss iters 2", 1, 11, low_pass_blur_iters2, &low_pass_blur_iters2);

#ifdef VISUALIZE
	int show_only_the_output = 1;
	vs_gui_add_switch("show only output", 1, &show_only_the_output);
#endif


	// Frame Loop
	while (1) {
		vs_frame_loop_control();

		load_alpha_into_registers(initial_alpha_LPF1, ALPHA_LPF1_WIDTH, LPF1_BITS);
		load_alpha_into_registers(initial_alpha_LPF2, ALPHA_LPF2_WIDTH, LPF2_BITS);

		// get the image and store it into analog register C
		get_image_store_it_in_C();

		// blur the image coming from C and move it into register C
		if (run_blur_1)
			blur_image_and_store_it_in_C();

		if (run_LPF1)
			apply_LPF_to_A_from_image_C(add_reminder);


		if (run_HPF)
			apply_HPF_to_C_from_C_and_LPF_from_A();


		if (run_blur_2)
			blur_image_2_and_store_it_in_D();

		if (run_LPF2)
			apply_LPF_to_B_from_image_D(add_reminder);

		if (run_subtraction)
			apply_subtraction();

#ifdef VISUALIZE
		if (vs_gui_is_on())
		{
			if (show_only_the_output)
			{
				scamp5_output_image(F, display_1);
			}
			else
			{
				scamp5_output_image(A, display_1);
				scamp5_output_image(C, display_2);
				scamp5_output_image(B, display_3);
				scamp5_output_image(F, display_4);
			}
		}
#endif
	}
	return 0;
}
*/
