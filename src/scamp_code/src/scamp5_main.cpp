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
 * 	-------------------------------------------------
 *  |						|						|
 *  |		Computation		|		   STO.1		|
 *  |						|						|
 *  |						|						|
 *  |						|						|
 *  |						|						|
 *  |						|						|
 *  |						|						|
 *  -------------------------------------------------
 *  |						|						|
 *  |		STO.2			|		   STO.3		|
 *  |						|						|
 *  |						|						|
 *  |						|						|
 *  |						|						|
 *  |						|						|
 *  |						|						|
 *  -------------------------------------------------
 *
 *
 */

struct low_pass_filter
{
	int no_bits;
	int value;
};


#include <scamp5.hpp>

using namespace SCAMP5_PE;

#define DREG_PE 	S0
#define DREG_STO1 	S1
#define DREG_STO2	S2
#define DREG_STO3   S3
#define BLUR_ITERATIONS 1
#define BLUR_ITERATIONS 2
#define LPF_FORMULATION_1

low_pass_filter LPF1 = { 5,
						 0b10000
};

low_pass_filter LPF2 = { 5,
						 0b10000
};


const inline void load_is_PE_into_S0()
{
	scamp5_load_pattern(S0, 0b11111110, 0b11111111, 0b11111110, 0b11111110);
}

const inline void load_is_ST1_into_S0()
{
	scamp5_load_pattern(S0, 0b11111110, 0b11111110, 0b11111110, 0b11111110);
}

const inline void load_is_ST1_into_S1()
{
	scamp5_load_pattern(S1, 0b11111110, 0b11111110, 0b11111110, 0b11111110);
}

const inline void load_is_ST2_into_S0()
{
	scamp5_load_pattern(S0, 0b11111111, 0b11111111, 0b11111110, 0b11111110);
}

const inline void load_is_ST2_into_S2()
{
	scamp5_load_pattern(S2, 0b11111111, 0b11111111, 0b11111110, 0b11111110);
}

const inline void load_is_ST3_into_S0()
{
	scamp5_load_pattern(S0, 0b11111111, 0b11111110, 0b11111110, 0b11111110);
}

const inline void load_is_ST3_into_S3()
{
	scamp5_load_pattern(S3, 0b11111111, 0b11111110, 0b11111110, 0b11111110);
}

const inline void move_the_image_for_A()
{
	scamp5_kernel_api::WHERE(DREG_PE);
		scamp5_kernel_api::bus(NEWS, A);

	scamp5_kernel_api::WHERE(DREG_STO1);
		scamp5_kernel_api::bus(A, XW);

	scamp5_kernel_api::WHERE(DREG_STO2);
		scamp5_kernel_api::bus(A, XN);
		scamp5_kernel_api::bus(NEWS, A);

    scamp5_kernel_api::WHERE(DREG_STO3);
    	scamp5_kernel_api::bus(A, XW);

	scamp5_kernel_api::ALL();
}

const inline void fill_A()
{
	scamp5_kernel_begin();
		move_the_image_for_A();
	scamp5_kernel_end();
}

void apply_lowpass_filter_incoming_A_previous_B_save_to_B(low_pass_filter &filter_param)
{
	int el = 1 << (filter_param.no_bits-1);

	while (el)
	{
		if (el & filter_param.value)
		{
			scamp5_kernel_begin();
				add(B, B, A);
				diva(A, E, F);
			scamp5_kernel_end();
		}
		else
		{
			scamp5_kernel_begin();
				diva(A, E, F);
			scamp5_kernel_end();
		}
		el >>= 1;
	}
}


int main(){

	// Initialization
	vs_init();
    vs_on_shutdown([&](){
    	vs_post_text("M0 shutdown\n");
    });

	// Setup Host GUI
	vs_gui_set_info(VS_M0_PROJECT_INFO_STRING);

	// create a normal-sized display in the Host App
    auto display_00 = vs_gui_add_display("display_00",0,0);

	// create a normal-sized display in the Host App
    auto display_01 = vs_gui_add_display("display_01",0,1);

	// create a normal-sized display in the Host App
    auto display_10 = vs_gui_add_display("display_10",1,0);

	// create a normal-sized display in the Host App
    auto display_11 = vs_gui_add_display("display_11",1,1);


    int apply_blur_1;
    vs_gui_add_slider("apply_blur_1", 0, 1, 0, &apply_blur_1);

    int apply_1st_hpf;
    vs_gui_add_slider("apply_1st_hpf", 0, 1, 0, &apply_1st_hpf);

    int apply_1st_lpf;
    vs_gui_add_slider("apply_1st_lpf", 0, 1, 0, &apply_1st_lpf);

    int apply_2nd_blur;
    vs_gui_add_slider("apply_2nd_blur", 0, 1, 0, &apply_2nd_blur);

    int apply_2nd_lpf;
    vs_gui_add_slider("apply_2nd_lpf", 0, 1, 0, &apply_2nd_lpf);

    auto slider_vxc = vs_gui_add_slider("vxc: ",0,4095,2900);
    vs_on_gui_update(slider_vxc,[&](int32_t new_value)
    {
    	vs_scamp5_configure_voltage(3, new_value);
    });

    auto lpf1 = vs_gui_add_slider("lpf1.value: ",0,32,LPF1.value);
    vs_on_gui_update(lpf1,[&](int32_t new_value)
    {
    	LPF1.value = new_value;
    });

    auto lpf2 = vs_gui_add_slider("lpf2.value: ",0,32,LPF2.value);
    vs_on_gui_update(lpf2,[&](int32_t new_value)
    {
    	LPF2.value = new_value;
    });

    int compute_opl;
    vs_gui_add_slider("compute_opl", 0, 1, 0, &compute_opl);

    /*
    auto slider_vdiff = vs_on_gui_update(vs_gui_add_slider("vdiff: ",0,800,scamp7_internal_voltage_setting[5]),[](int32_t new_value){
    	scamp7_set_internal_dac(5,new_value);
    	scamp7_internal_voltage_setting[5] = new_value;
    	vs_post_text("vdiff -> %d\n",new_value);
    	});
    */

    scamp5_kernel_begin();
    	ALL();
    		scamp5_load_in(A, 0);
    		scamp5_load_in(B, 0);
    		scamp5_load_in(C, 0);
    		scamp5_load_in(D, 0);
    		scamp5_load_in(E, 0);
    		scamp5_load_in(F, 0);
    scamp5_kernel_end();


    bool first_pass = true;

	// Frame Loop
    while(1)
    {
        vs_frame_loop_control();

        // load the status registers
        load_is_PE_into_S0();
        load_is_ST1_into_S1();
        load_is_ST2_into_S2();
        load_is_ST3_into_S3();

        // capture the image into A, and F, but only where the S0 is true ( is PE)
		scamp5_get_image(A, F);


		// apply blur if requested
		if (apply_blur_1)
		{
			scamp5_kernel_begin();
				move_the_image_for_A();

				SET(R1, R2);
				gauss(A, A, BLUR_ITERATIONS);

			scamp5_kernel_end();
		}

		if (apply_1st_hpf)
		{
			// compute the HPF firstly by
			// doing e = x[i] - x[i-1]
			// Incoming = PE.A - STO1.B

			scamp5_kernel_begin();
				WHERE(DREG_STO1);
					bus(NEWS, B);

				WHERE(DREG_PE);
					bus(B, XE);

					sub(F, A, B);
					mov(B, F);

					bus(NEWS, A);

				WHERE(DREG_STO1);
					bus(B, XW);

				WHERE(DREG_PE);
					mov(A, B);

			scamp5_kernel_end();
		}

		if (apply_1st_lpf)
		{
			if (first_pass)
			{
				scamp5_kernel_begin();
					WHERE(DREG_PE);
						mov(B, A);
				scamp5_kernel_end();
			}
			else
			{
				// PE.A is incoming and PE.B is the previous LPF
				// the idea is to compute
				// PE.B = PE.B + alpha * (PE.A - PE.B)
				scamp5_kernel_begin();
					WHERE(DREG_STO1);
						bus(NEWS, C);

					WHERE(DREG_PE);
						bus(B, XE);

						sub(A, A, B);
						diva(A, E, F);
				scamp5_kernel_end();
			}

			// now compute the LPF on the incoming data from A
			// formulation for the LPF we use
			// LPF [ST1.C] := LPF [ST1.C] + alpha * (LPF [ST1.C] - Incoming [PE.A])
			apply_lowpass_filter_incoming_A_previous_B_save_to_B(LPF1);

			// move the PE.B back to STO1.C
			scamp5_kernel_begin();
				bus(NEWS, B);

				WHERE(DREG_STO1);
					bus(C, XW);

				WHERE(DREG_PE);
					mov(A, B);
					mov(C, B);
			scamp5_kernel_end();
		}

		// apply second blur
		fill_A();

		if (apply_2nd_blur)
		{
			scamp5_kernel_begin();
				ALL();
				SET(R1, R2);
				gauss(A, A, BLUR_ITERATIONS);

			scamp5_kernel_end();
		}


		if (apply_2nd_lpf)
		{
			// incoming is in A, and we apply it to B
			if (first_pass)
			{
				scamp5_kernel_begin();
					WHERE(DREG_PE);
						mov(B, A);
				scamp5_kernel_end();
			}
			else
			{
				// PE.A is incoming and PE.B is the previous LPF
				// the idea is to compute
				// PE.B = PE.B + alpha * (PE.A - PE.B)
				scamp5_kernel_begin();
					WHERE(DREG_STO1);
						bus(NEWS, D);

					WHERE(DREG_PE);
						bus(B, XE);

						sub(A, A, B);
						diva(A, E, F);
				scamp5_kernel_end();
			}

			apply_lowpass_filter_incoming_A_previous_B_save_to_B(LPF2);

			// move the PE.B back to STO1.D
			scamp5_kernel_begin();
				bus(NEWS, B);

				WHERE(DREG_STO1);
					bus(D, XW);

				WHERE(DREG_PE);
					mov(A, B);
			scamp5_kernel_end();

		}

		if (compute_opl)
		{
			scamp5_kernel_begin();
				sub(A, A, C);
			scamp5_kernel_end();
		}


		fill_A();


		first_pass = false;

		if(vs_gui_is_on()){
			scamp5_output_image(A, display_00);
			//scamp5_output_image(B, display_01);
			//scamp5_output_image(S2, display_10);
			//scamp5_output_image(S3, display_11);

            vs_post_text("(%lu) %s\n",VS_LC,scamp5_verbose_str());
		}
    }

	return 0;
}
