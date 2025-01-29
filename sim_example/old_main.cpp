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


#define ALPHA_LPF1_WIDTH  		   4

const uint8_t LOW_PASS_FILTER1_ITERS = 1;
const uint8_t initial_alpha_LPF1 = 0b0010;
dreg_t LPF1_BITS[ALPHA_LPF1_WIDTH] = { SCAMP5_PE::R3, SCAMP5_PE::R4, SCAMP5_PE::R5, SCAMP5_PE::R6 };

using namespace SCAMP5_PE;

bool LPF1_first = true;
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
 // temporary : F
void get_image_store_it_in_C()
{
	scamp5_kernel_begin();
	get_image(C, F);
	scamp5_kernel_end();
}

// temporary : F
void blur_image_and_store_it_in_C()
{
	scamp5_kernel_begin();
	SET(R1, R2);
	gauss(F, C, LOW_PASS_FILTER1_ITERS);
	mov(C, F);
	scamp5_kernel_end();
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
void apply_LPF_to_B_from_image_C()
{
	if (LPF1_first)
	{
		scamp5_kernel_begin();
		mov(A, C);
		scamp5_kernel_end();
		load_alpha_into_registers(initial_alpha_LPF1, ALPHA_LPF1_WIDTH, LPF1_BITS);
		LPF1_first = false;
	}

	// store the C - A in D
	scamp5_kernel_begin();
	sub(D, C, A);
	scamp5_kernel_end();

	for (int i = 0; i < ALPHA_LPF1_WIDTH; ++i)
	{
		scamp5_dynamic_kernel_begin();
		diva(D, E, F);
		WHERE(LPF1_BITS[i]);
		add(C, C, D);
		ALL();
		scamp5_dynamic_kernel_end();
	}
}


int main() {

	vs_init();

	vs_gui_set_info(VS_M0_PROJECT_INFO_STRING);

#ifdef VISUALIZE
	auto display_1 = vs_gui_add_display("image", 0, 0, 1);
	auto display_2 = vs_gui_add_display("blurred", 0, 1, 1);
	auto display_3 = vs_gui_add_display("low_passed", 0, 2, 1);
#endif

	// Frame Loop
	while (1) {
		vs_frame_loop_control();

		// get the image and store it into analog register C
		get_image_store_it_in_C();


#ifdef VISUALIZE
		scamp5_kernel_begin();
		mov(E, C);
		scamp5_kernel_end();
		if (vs_gui_is_on()) {
			scamp5_output_image(E, display_1);
		}
#endif

		// blur the image coming from C and move it into register C
		blur_image_and_store_it_in_C();

#ifdef VISUALIZE
		if (vs_gui_is_on()) {
			scamp5_output_image(C, display_2);
		}
#endif

		apply_LPF_to_B_from_image_C();

#ifdef VISUALIZE
		if (vs_gui_is_on())
		{
			scamp5_output_image(B, display_3);
		}
#endif
	}
	return 0;
}

*/