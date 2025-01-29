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
#define SIMULATION
#include <scamp5.hpp>

#define ARED_BLURRED_IMG_REG		A
#define AREG_LPF1_REG				B
#define AREG_LPF2_REG				C
#define DREG_IS_PE					R5
#define DREG_IS_STORAGE				R6

bool first_run = true;


struct LPF_fixed
{
	int no_of_bits_for_alpha = 4;
	int alpha = 0b0010;
};

using namespace SCAMP5_PE;


int main()
{

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

	int run_blur_1 = 0;
	vs_gui_add_switch("run_blur_1", 0, &run_blur_1);

	int run_avg_1 = 0;
	vs_gui_add_switch("run_avg_1", 0, &run_avg_1);


	LPF_fixed lpf1 = { 4,			// width of alpha
					   0b0001		// alpha value
	};

	LPF_fixed lpf2 = { 4,
					   0b0011,
	};

	vs_gui_add_slider("alpha1_value", 0, 255, 2, &lpf1.alpha);
	vs_gui_add_slider("alpha1_bits", 1, 6, 4, &lpf1.no_of_bits_for_alpha);

	vs_gui_add_slider("alpha2_value", 0, 255, 2, &lpf2.alpha);
	vs_gui_add_slider("alpha2_bits", 1, 6, 4, &lpf2.no_of_bits_for_alpha);



#ifdef VISUALIZE_DEBUG
	auto display_1 = vs_gui_add_display("image", 0, 0, 1);
	auto display_2 = vs_gui_add_display("LPF", 0, 1, 1);
	auto display_3 = vs_gui_add_display("HPF", 1, 0, 1);
	auto display_4 = vs_gui_add_display("DREG", 1, 1, 1);
#endif


	// Frame Loop
	while (1) {
		vs_frame_loop_control();

		// load pattern 
		scamp5_load_pattern(DREG_IS_STORAGE, 0b0000001, 0b00000000, 0b11111110, 0b11111111);
		scamp5_kernel_begin();
			NOT(DREG_IS_PE, DREG_IS_STORAGE);
		scamp5_kernel_end();
		
		// load image to A 
		scamp5_get_image(A, F, 1);

		if (run_blur_1)
		{
			scamp5_kernel_begin();
			SET(R1, R2);
				gauss(A, A, 1);
			CLR(R1, R2);
			scamp5_kernel_end();
		}

		// compute the low pass filter for the blurred image coming from A 
		// this can be done by loading a value from B from storage where the LPF is stored 
		if (first_run)
		{
			scamp5_kernel_begin();
				WHERE(DREG_IS_PE);
					mov(B, A);
			scamp5_kernel_end();
		}
		else
		{
			scamp5_kernel_begin();
			WHERE(DREG_IS_STORAGE);
				bus(NEWS, AREG_LPF1_REG);

			// take the 
			WHERE(DREG_IS_PE);
				bus(B, XS);
			scamp5_kernel_end();
		}

		// now compute the Low Pass value and update B using it 
		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				sub(C, A, B);
				diva(C, E, F);
		scamp5_kernel_end();

		int MSB = 1 << (lpf1.no_of_bits_for_alpha - 1);

		while (MSB)
		{
			if (MSB & lpf1.alpha)
			{
				scamp5_kernel_begin();
					WHERE(DREG_IS_PE);
					add(B, B, C);
				scamp5_kernel_end();
			}

			scamp5_kernel_begin();
				WHERE(DREG_IS_PE);
					diva(C, E, F);
			scamp5_kernel_end();

			MSB >>= 1;
		}

		// th updated LPF has to be now saved into the storage by following
		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);	
				bus(NEWS, B);
			
			WHERE(DREG_IS_STORAGE);
				bus(B, XN);
		scamp5_kernel_end();

		// Compute the HPF, by subtracting C from A 
		scamp5_kernel_begin();
			sub(A, A, B);
		scamp5_kernel_end();

		first_run = false;

#ifdef VISUALIZE_DEBUG
		if (vs_gui_is_on())
		{
			scamp5_output_image(A, display_1);
			scamp5_output_image(B, display_2);
			scamp5_output_image(C, display_3);
			scamp5_output_image(D, display_4);
		}
#endif

	}
	return 0;
}
