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
#define SUBTRACT_OFFSSET			-127
#define ADD_OFFSET					127

bool first_run = true;
int MSB;

struct LPF_fixed
{
	int no_of_bits_for_alpha = 4;
	int alpha = 0b0010;
};

using namespace SCAMP5_PE;


void load_STO_AREG_to_PE_AREG(areg_t source, areg_t destination)
{
	scamp5_dynamic_kernel_begin();
	WHERE(DREG_IS_STORAGE);
	bus(NEWS, source);

	// take the
	WHERE(DREG_IS_PE);
	bus(destination, XS);
	scamp5_dynamic_kernel_end();
}

void store_PE_AREG_to_STO_AREG(areg_t source, areg_t destination)
{
	scamp5_dynamic_kernel_begin();
		WHERE(DREG_IS_PE);
		bus(NEWS, source);

		WHERE(DREG_IS_STORAGE);
		bus(destination, XN);
	scamp5_dynamic_kernel_end();
}

void compute_LPF(areg_t incoming_map,
	areg_t LPF_prev,
	areg_t incoming_sub_LPF_prev,
	areg_t div1_areg,
	areg_t div2_areg,
	LPF_fixed& lpf_to_apply
)
{
	// now compute the Low Pass value and update B using it
	scamp5_dynamic_kernel_begin();
	WHERE(DREG_IS_PE);
	sub(incoming_sub_LPF_prev, incoming_map, LPF_prev);
	diva(incoming_sub_LPF_prev, div1_areg, div2_areg);
	scamp5_dynamic_kernel_end();

	int MSB = 1 << (lpf_to_apply.no_of_bits_for_alpha - 1);

	while (MSB)
	{
		if (MSB & lpf_to_apply.alpha)
		{
			scamp5_dynamic_kernel_begin();
			WHERE(DREG_IS_PE);
			add(LPF_prev, LPF_prev, incoming_sub_LPF_prev);
			scamp5_dynamic_kernel_end();
		}

		scamp5_kernel_begin();
		WHERE(DREG_IS_PE);
		diva(incoming_sub_LPF_prev, div1_areg, div2_areg);
		scamp5_kernel_end();

		MSB >>= 1;
	}
}


void compute_full_LPF_using_STO(areg_t STO_register_with_lpf,
	areg_t PE_register,
	areg_t incoming,
	areg_t incoming_sub_LPF_prev,
	areg_t div1_areg,
	areg_t div2_areg,
	LPF_fixed& lpf)
{

	// if we have first run, we move the mapso we operate on this
	if (first_run)
	{
		scamp5_dynamic_kernel_begin();
		WHERE(DREG_IS_PE);
		mov(PE_register, incoming);
		scamp5_dynamic_kernel_end();
	}
	else
	{
		load_STO_AREG_to_PE_AREG(STO_register_with_lpf, // areg where taken from PE (STO.areg -> NEWS)
			PE_register);		    // areg where we take that value from STO (south news -> B)
	}

	// compute the low pass filter
	compute_LPF(incoming,					// incoming map
		PE_register,				// where the previous map is
		incoming_sub_LPF_prev,		// where the incoming - previous map will be saved
		div1_areg,					// where the 1st intermediate from division will be applied
		div2_areg,					// where the 2nd intermediate from division will be applied
		lpf);


	store_PE_AREG_to_STO_AREG(PE_register,				// areg where copied from PE (source -> NEWS)
							  STO_register_with_lpf);	// areg where to store back to STO (north NS -> B)


}

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

	LPF_fixed lpf3 = { 4,
					   0b0010,
	};

	LPF_fixed lpf4 = { 4,
					  0b0010
	};

	LPF_fixed lpf5 = { 4,
					  0b0001
	};


	int G0 = 0;
	int Ibias = 0;
	int threshold = 0;

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


#ifdef VISUALIZE_DEBUG
	auto display_1 = vs_gui_add_display("image", 0, 0, 1);
	auto display_2 = vs_gui_add_display("LPF", 0, 1, 1);
	auto display_3 = vs_gui_add_display("HPF", 0, 2, 1);
	auto display_4 = vs_gui_add_display("membranes", 1, 0, 1);
	auto display_5 = vs_gui_add_display("events", 1, 1, 1);
#endif


	// Frame Loop
	while (1) {
		vs_frame_loop_control();

		// load pattern
		scamp5_load_pattern(DREG_IS_STORAGE, 0b0000001, 0b00000000, 0b11111110, 0b11111111);
		scamp5_kernel_begin();
		NOT(DREG_IS_PE, DREG_IS_STORAGE);
		scamp5_kernel_end();


		/*
		 * DREG_IS_PE (R5) 		: masked least significant bit, odds
		 * DREG_IS_STORAGE (R6)	: inverse of R5
		 *
		 * load image to A
		 */
		scamp5_get_image(A, F, 1);

		if (run_blur_1)
		{
			scamp5_kernel_begin();
			SET(R1, R2);
			gauss(A, A, 1);
			CLR(R1, R2);
			scamp5_kernel_end();
		}


		compute_full_LPF_using_STO(AREG_LPF1_REG,					// where we take the LPF register from storage cells
			B,								// where we save the loaded storage LPF map in PE
			A,								// register representing the incoming map
			C,								// where we store the incoming -
			E,								// where the first intermediate result will be from division
			F,								// where the second intermediate result will be from division
			lpf1);							// parameters for low pass 1

		// Compute the HPF, by subtracting B from A
		// move this map to C, as it will be used later
		scamp5_kernel_begin();
		sub(A, A, B);
		mov(C, A);
		scamp5_kernel_end();

		/*
		// for diffusion of the spatial filter move it to A
		scamp5_kernel_begin();
			CLR(R1);
			SET(R2);
			gaussh(A,A,1);

			SET(R1);
			CLR(R2);
			gaussv(A, A, 2);
		scamp5_kernel_end();

		// Now load from STO.C which contains previous map for LPF2 to PE.B
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
					bus(NEWS, C);

				WHERE(DREG_IS_PE);
					bus(B, XS);
			scamp5_kernel_end();
		}

		// compute the lpf2 on A with a map stored in B
		MSB = 1 << (lpf1.no_of_bits_for_alpha - 1);

		// PE.A := (PE.A - PE.B)
		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				sub(A, A, B);
				diva(A, E, F);
		scamp5_kernel_end();


		while (MSB)
		{
			if (MSB & lpf2.alpha)
			{
				scamp5_kernel_begin();
					add(B, B, A);
				scamp5_kernel_end();
			}

			MSB >>= 1;
			scamp5_kernel_begin();
				diva(A, E, F);
			scamp5_kernel_end();
		}

		// transmit back the PE.B to STO.C
		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				bus(NEWS, B);
			WHERE(DREG_IS_STORAGE);
				bus(C, XN);
		scamp5_kernel_end();


		// compute the OPL by subtracting B and C
		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				sub(A, B, C);
		scamp5_kernel_end();

		// OPL is now saved in A, compute the VBip
		// by doing a LPF on OPL

		// load the LPF3 from STO.D to PE.B
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
					bus(NEWS, D);

				WHERE(DREG_IS_PE);
					bus(B, XS);
			scamp5_kernel_end();
		}

		// compute the LPF 3 (adaptive)
		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				sub(C, A, B);
				diva(C, E, F);
		scamp5_kernel_end();

		// TODO: IMPLEMENT FEEDBACK
		MSB = 1 << (lpf3.no_of_bits_for_alpha - 1);

		while(MSB)
		{
			if (MSB & lpf3.alpha)
			{
				scamp5_kernel_begin();
				WHERE(DREG_IS_PE);
					add(B, B, C);
				scamp5_kernel_end();
			}
			MSB >>= 1;

			scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				diva(C, E, F);
			scamp5_kernel_end();
		}
		// A stores OPL, B stores the VBip, now the C := B - A will be values that will be converted into ganglion current

		scamp5_kernel_begin();
			sub(C, A, B);
		scamp5_kernel_end();

		// TODO COMPLEX equation, computing
		// A := G0 + lambda * V_Bip^2
		// firstly, store PE.B back to STO.D
		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				bus(NEWS, B);

			WHERE(DREG_IS_STORAGE);
				bus(D, XN);
		scamp5_kernel_end();

		// compute an approximate : A := G0 + abs(V_Bip)
		scamp5_in(F, G0);

		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
			abs(B, B);

			WHERE(DREG_IS_PE); // abs changes the flag to all !
			add(B, B, F);
			mov(A, B);
		scamp5_kernel_end();

		// load the 4th low pass filter from STO.E to PE.B
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
				mov(NEWS, E);

			WHERE(DREG_IS_PE);
				mov(B, XS);
			scamp5_kernel_end();
		}

		// compute the 4th low pass, using value from A to B
		int MSB  = 1 << (lpf4.no_of_bits_for_alpha - 1);

		// store the delta in A
		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				sub(A, A, B);
				diva(A, E, F);
		scamp5_kernel_end();

		while (MSB)
		{
			if (MSB & lpf4.alpha)
			{
				scamp5_kernel_begin();
					WHERE(DREG_IS_PE);
						add(B, B, A);
				scamp5_kernel_end();
			}


			MSB >>= 1;
			scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				diva(A, E, F);
			scamp5_kernel_end();
		}

		// store the PE.B to STO. E

		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				bus(NEWS, B);
			WHERE(DREG_IS_STORAGE);
				bus(E, XN);
		scamp5_kernel_end();


		// TODO: Now that this is computed, need to implement the



		// Now that the Feedback is completed, apply the U function
		// which will be approximated as max(0, C) + Ibias
		scamp5_in(F, SUBTRACT_OFFSSET);
		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				add(C, C, F);
		scamp5_kernel_end();

		scamp5_in(F, ADD_OFFSET);
		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				add(C, C, F);
		scamp5_kernel_end();


		scamp5_in(F, Ibias);
		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				add(C, C, F);
		scamp5_kernel_end();

		// load the 5th LPF
		if (first_run)
		{
			scamp5_kernel_begin();
				WHERE(DREG_IS_PE);
				mov(C, D);
			scamp5_kernel_end();
		}

		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
				sub(A, C, D);
				diva(A, E, F);
		scamp5_kernel_end();


		MSB = 1 << (lpf5.no_of_bits_for_alpha - 1);

		while (MSB)
		{
			if (lpf5.alpha & MSB)
			{
				scamp5_kernel_begin();
					WHERE(DREG_IS_PE);
						add(E, E, A);
				scamp5_kernel_end();
			}

			MSB >>= 1;
			scamp5_kernel_begin();
				WHERE(DREG_IS_PE);
					diva(A, E, F);
			scamp5_kernel_end();
		}

		// PE.E stores now the voltages on the membrane for LI neuro
		// we set the threshold now that will be subtracted from E, to trasnmit the events

		scamp5_in(F, threshold);

		scamp5_kernel_begin();
			WHERE(DREG_IS_PE);
			sub(E, E, F);

			CLR(R0);
			where(E);

			MOV(R6, FLAG);
			NOT(R7, R6);
			MOV(R6, R7);

			ALL();
		scamp5_kernel_end();
		// FLAG should have now contain the elements that fired
		*/

		first_run = false;

#ifdef VISUALIZE_DEBUG
		if (vs_gui_is_on())
		{
			scamp5_output_image(A, display_1);
			scamp5_output_image(B, display_2);
			//scamp5_output_image(C, display_3);
			//scamp5_output_image(E, display_4);
			//scamp5_output_image(R6, display_5);
		}
#endif

	}
	return 0;
}
