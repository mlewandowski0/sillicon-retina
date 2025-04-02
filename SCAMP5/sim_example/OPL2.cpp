/*************************************************************************
 * SCAMP Vision Chip Development System Library
 *------------------------------------------------------------------------
 * Copyright (c) 2020 The University of Manchester. All Rights Reserved.
 *
 *************************************************************************/

/*
  // 256 x 128

#define VISUALIZE_DEBUG
//#define SIMULATION
#define DREG_IS_PROCESSING_EL S0

#include <scamp5.hpp>


bool first_run = true;


struct LPF_fixed
{
	int no_of_bits_for_alpha = 4;
	int alpha = 0b0010;
};

using namespace SCAMP5_PE;


void load_pattern(dreg_t imm = R11)
{
	scamp5_load_pattern(DREG_IS_PROCESSING_EL, 0b0000001, 0b00000000, 0b11111110, 0b11111111);
	scamp5_dynamic_kernel_begin();
	NOT(imm, DREG_IS_PROCESSING_EL);
	MOV(DREG_IS_PROCESSING_EL, imm);
	scamp5_dynamic_kernel_end();
}

// Intermediate : F
void load_into_storage(int8_t value, areg_t reg, areg_t imm = F)
{
	scamp5_dynamic_kernel_begin();
	mov(imm, reg);
	scamp5_dynamic_kernel_end();

	scamp5_load_in(reg, value);
	scamp5_dynamic_kernel_begin();
	WHERE(R3);
	mov(reg, imm);
	scamp5_dynamic_kernel_end();
}

void load_from_storage_to_PE(areg_t dest, areg_t source, dreg_t digital_imm = R11)
{
	scamp5_dynamic_kernel_begin();
	// select not the processing elements
	NOT(digital_imm, DREG_IS_PROCESSING_EL);
	WHERE(digital_imm);
	bus(NEWS, source);

	WHERE(DREG_IS_PROCESSING_EL);
	bus(dest, XS);

	ALL();
	CLR(R1, R2, R3, R4);
	scamp5_dynamic_kernel_end();
}

void save_from_PE_to_storage(areg_t dest, areg_t source, dreg_t digital_imm = R11)
{
	scamp5_dynamic_kernel_begin();
	// select the processing elements
	NOT(digital_imm, DREG_IS_PROCESSING_EL);
	WHERE(DREG_IS_PROCESSING_EL);
	bus(NEWS, source);

	WHERE(digital_imm);
	bus(dest, XN);

	ALL();
	CLR(R1, R2, R3, R4);
	scamp5_dynamic_kernel_end();
}

void load_image_into_A(areg_t imm = F, int gain = 1)
{
	scamp5_dynamic_kernel_begin();
	scamp5_get_image(A, imm, gain);
	scamp5_dynamic_kernel_end();
}

void blur_register(areg_t source, int iterations = 1)
{
	scamp5_dynamic_kernel_begin();
	SET(R1, R2);
	gauss(source, source, iterations);

	scamp5_dynamic_kernel_end();
}

void blur_register_256x128(areg_t source, int iterations = 1)
{
	scamp5_dynamic_kernel_begin();
	ALL();
	CLR(R1);
	SET(R2);
	gaussh(source, source, iterations);

	SET(R1);
	CLR(R2);
	gaussv(source, source, 2 * iterations);

	CLR(R1);
	scamp5_dynamic_kernel_end();
}

void average_pixels(areg_t source)
{
	load_from_storage_to_PE(B, source);

	scamp5_kernel_begin();
	WHERE(DREG_IS_PROCESSING_EL);
	diva(A, E, F);
	diva(B, E, F);
	add(A, A, B);
	ALL();
	scamp5_kernel_end();
}

void compute_low_pass_fixed(areg_t incoming,
	areg_t LPF_areg,
	areg_t delta_storage,
	LPF_fixed& lpf,
	areg_t diva1_imm = E,
	areg_t diva2_imm = F)
{
	scamp5_dynamic_kernel_begin();
	// select only elements in where the computation happens, i.e. every row
	WHERE(DREG_IS_PROCESSING_EL);

	// compute the delta
	sub(delta_storage, incoming, LPF_areg);

	// compute the delta := 0.5 * delta
	diva(delta_storage, diva1_imm, diva2_imm);
	scamp5_dynamic_kernel_end();

	uint8_t bits = lpf.no_of_bits_for_alpha;
	uint8_t MSB = 1 << (bits - 1);

	while (MSB)
	{
		if (lpf.alpha & MSB)
		{
			scamp5_dynamic_kernel_begin();
			WHERE(DREG_IS_PROCESSING_EL);
			add(LPF_areg, LPF_areg, delta_storage);
			scamp5_dynamic_kernel_end();
		}
		bits -= 1;
		MSB >>= 1;


		// compute the delta := 0.5 * delta
		scamp5_dynamic_kernel_begin();
		WHERE(DREG_IS_PROCESSING_EL);
		diva(delta_storage, diva1_imm, diva2_imm);
		scamp5_dynamic_kernel_end();
	}

	scamp5_kernel_begin();
	ALL();
	scamp5_kernel_end();
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
					   0b0010		// alpha value
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

		load_pattern();
		//scamp5_dynamic_kernel_begin();
		//	REFRESH(DREG_IS_PROCESSING_EL);
		//scamp5_dynamic_kernel_end();

		// load the image into A, immediate F


		load_image_into_A(F);

		if (run_blur_1)
		{
			// blur images for A
			blur_register(A);
		}


		if (run_avg_1)
		{
			// average the pixels
			average_pixels(A);
		}


		 // load the Low pass filter
		if (!first_run)
			load_from_storage_to_PE(B, B);	// source = B [storage], destination = B [PE]
		else
		{
			scamp5_kernel_begin();
			WHERE(DREG_IS_PROCESSING_EL);
			mov(B, A);
			ALL();
			scamp5_kernel_end();
		}




		 // compute the first LPF
		compute_low_pass_fixed(A,		// take the value from A to filter it : in this context this is blurred image
			B,							// where is the previous history of the LPF is stored ; this was loaded from previous step from storage to B
			C,							// register where the A - B will be stored
			lpf1,						// parameters of first low pass
			E,							// diva immediate 1
			F);							// diva immediate 2





		 // write back the LPF value
		save_from_PE_to_storage(B, B);	// source = B [PE], destination = B [storage



		 // compute the HPF
		scamp5_kernel_begin();
		WHERE(DREG_IS_PROCESSING_EL);
		sub(A, A, B);
		mov(D, A);
		ALL();
		scamp5_kernel_end();



		 // stores the high pass in C, and copy it in A


		 // store the image
		save_from_PE_to_storage(A, A);

		// blur the HPF from A
		blur_register_256x128(A, 2);
		// load second LPF from C [storage] to B[PE]
		if (!first_run)
			load_from_storage_to_PE(B, C);
		else
		{
			scamp5_kernel_begin();
			WHERE(DREG_IS_PROCESSING_EL);
			mov(B, A);
			ALL();
			scamp5_kernel_end();
		}


		// compute the first LPF
		compute_low_pass_fixed(A,		// take the value from A to filter it : in this context this is blurred image
			B,							// where is the previous history of the LPF is stored ; this was loaded from previous step from storage to B
			A,							// register where the A - B will be stored ; discard the blurred image, it is no longer needed
			lpf2,						// parameters of first low pass
			E,							// diva immediate 1
			F);							// diva immediate 2

		// write back the LPF value
		save_from_PE_to_storage(B, C);	// source = C [PE], destination = C [storage]


		// Compute the OPL
		scamp5_kernel_begin();
		WHERE(DREG_IS_PROCESSING_EL);
		sub(A, D, B);
		ALL();
		scamp5_kernel_end();



		save_from_PE_to_storage(A, A);	// source = C [PE], destination = C [storage]
		save_from_PE_to_storage(D, D);	// source = C [PE], destination = C [storage]

		if (first_run)
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
*/