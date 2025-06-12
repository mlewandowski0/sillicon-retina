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
#define DREG_STO3	S3
#define VBIP_GREATER_THAN_64 S4
#define VBIP_GREATER_THAN_32 S5
#define VBIP_GREATER_THAN_16 S6

#define BLUR_ITERATIONS 1
#define BLUR_ITERATIONS_2 2
#define BLUR_ITERATIONS_BIPOLAR 1

#define LPF_FORMULATION_1

#define BIPOLAR_REGISTER_STO1 E
#define BIPOLAR_LPF_REGISTER_STO2 B


low_pass_filter LPF1 = { 5,
						 0b10000
};

low_pass_filter LPF2 = { 5,
						 0b10000
};

low_pass_filter lambda_A = { 5,
						 	   0b10000
};

low_pass_filter LPF3 = { 4,
						 0b1000
};

low_pass_filter alpha_fixed = { 5,
						 	   0b11000
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

const inline void move_the_image_for_F()
{
	scamp5_kernel_api::WHERE(DREG_PE);
		scamp5_kernel_api::bus(NEWS, F);

	scamp5_kernel_api::WHERE(DREG_STO1);
		scamp5_kernel_api::bus(F, XW);

	scamp5_kernel_api::WHERE(DREG_STO2);
		scamp5_kernel_api::bus(F, XN);
		scamp5_kernel_api::bus(NEWS, F);

    scamp5_kernel_api::WHERE(DREG_STO3);
    	scamp5_kernel_api::bus(F, XW);

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


    int apply_blur_1=0;
    vs_gui_add_slider("apply_blur_1", 0, 1, apply_blur_1, &apply_blur_1);

    int apply_1st_hpf=0;
    vs_gui_add_slider("apply_1st_hpf", 0, 1, apply_1st_hpf, &apply_1st_hpf);

    int apply_1st_lpf=0;
    vs_gui_add_slider("apply_1st_lpf", 0, 1, apply_1st_lpf, &apply_1st_lpf);

    int apply_2nd_blur=0;
    vs_gui_add_slider("apply_2nd_blur", 0, 1, apply_2nd_blur, &apply_2nd_blur);

    int apply_2nd_lpf=0;
    vs_gui_add_slider("apply_2nd_lpf", 0, 1, apply_2nd_lpf, &apply_2nd_lpf);

    int apply_3rd_lpf=0;
    vs_gui_add_slider("apply_3rd_lpf", 0, 1, apply_3rd_lpf, &apply_3rd_lpf);

    int compute_opl=0;
    vs_gui_add_slider("compute_opl", 0, 1, compute_opl, &compute_opl);

    int compute_approximation=0;
    vs_gui_add_slider("compute approximation", 0, 1, compute_approximation, &compute_approximation);

    int compute_blur_for_approximation=0;
    vs_gui_add_slider("compute blur for approximation", 0, 1, compute_blur_for_approximation, &compute_blur_for_approximation);

    int add_leftovers_to_feedback=0;
    vs_gui_add_slider("add leftovers to feedback", 0, 1, add_leftovers_to_feedback, &add_leftovers_to_feedback);

    int compute_feedback_phase2=0;
    vs_gui_add_slider("compute feedback phase2", 0, 1, compute_feedback_phase2, &compute_feedback_phase2);


    int g_A=0;
    vs_gui_add_slider("g_{A}^{0", -128, 127, g_A, &g_A);


    auto slider_vxc = vs_gui_add_slider("vxc: ",0,4095,2900);
    vs_on_gui_update(slider_vxc,[&](int32_t new_value)
    {
    	vs_scamp5_configure_voltage(3, new_value);
    });

    auto lpf1 = vs_gui_add_slider("lpf1.value: ",1,31,LPF1.value);
    vs_on_gui_update(lpf1,[&](int32_t new_value)
    {
    	LPF1.value = new_value;
    });

    auto lpf2 = vs_gui_add_slider("lpf2.value: ",1,31,LPF2.value);
    vs_on_gui_update(lpf2,[&](int32_t new_value)
    {
    	LPF2.value = new_value;
    });

    auto fm_1 = vs_gui_add_slider("lambda_A.value: ",1,31,lambda_A.value);
    vs_on_gui_update(fm_1,[&](int32_t new_value)
    {
    	lambda_A.value = new_value;
    });

    auto lpf3 = vs_gui_add_slider("lpf3.value: ",1,16,LPF3.value);
    vs_on_gui_update(lpf3,[&](int32_t new_value)
    {
    	LPF3.value = new_value;
    });

    auto fm_2 = vs_gui_add_slider("alpha_fixed.value: ",1,31,alpha_fixed.value);
    vs_on_gui_update(fm_2,[&](int32_t new_value)
    {
    	alpha_fixed.value = new_value;
    });


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
				WHERE(DREG_PE);
					sub(A, A, C);
			scamp5_kernel_end();
		}


		if (compute_approximation)
		{
			// Zero out B
			scamp5_load_in(F, 0);
			scamp5_kernel_begin();
				WHERE(DREG_PE);
					mov(B, F);
			scamp5_kernel_end();

			// move from storage 1 cell the Bipolar voltage (stored in STO1.E) to PE.D and store
			// absolute value of that voltage to PE.C
			scamp5_kernel_begin();
				WHERE(DREG_STO1);
					bus(NEWS, BIPOLAR_REGISTER_STO1);

				WHERE(DREG_PE);
					bus(D, XE);
					mov(F, D);

				// take the absolute value of F and store it in C
					abs(F, F);
				WHERE(DREG_PE);
					mov(C, F);

			scamp5_kernel_end();

			int val = 1 << (lambda_A.no_bits - 1);
			while (val)
			{
				// multiply D by alpha where alpha is b_1.b_2b_3b_4_b_5
				if (val & lambda_A.value)
				{
					scamp5_kernel_begin();
						add(B, B, C);
						diva(C, E, F);
					scamp5_kernel_end();
				}
				else
				{
					scamp5_kernel_begin();
						diva(D, E, F);
					scamp5_kernel_end();
				}

				val >>= 1;
			}
			// load the g_A^0 to F
			scamp5_load_in(F, g_A);

			// add F to B which currently stores the alpha * |V_bip| and store in C
			scamp5_kernel_begin();
				add(C, B, F);
			scamp5_kernel_end();

			// by this point
			// PE.A : I_{OPL}
			// PE.B : lambda_A * |V_bip|
			// PE.C : g_A + lambda_A * |V_bip|
			// PE.D : V_bip

			//
			if (first_pass)
			{
				scamp5_kernel_begin();
					WHERE(DREG_PE);
						mov(B, C);
				scamp5_kernel_end();
			}
			// transmit the LPF values from STO2.B to PE.B
			else
			{
				scamp5_kernel_begin();
					WHERE(DREG_STO2);
						bus(NEWS, BIPOLAR_LPF_REGISTER_STO2);

					WHERE(DREG_PE);
						bus(B, XS);
				scamp5_kernel_end();
			}

			// compute the LPF value by
			// B := B + alpha *(C - B)

			scamp5_kernel_begin();
				WHERE(DREG_PE);
					sub(C, C, B);
					diva(C, E, F);
			scamp5_kernel_end();

			val = 1 << (LPF3.no_bits - 1);

			while (val)
			{
				if (val & LPF3.value)
				{
					scamp5_kernel_begin();
						add(B, B, C);
						diva(C, E, F);
					scamp5_kernel_end();
				}
				else
				{
					scamp5_kernel_begin();
						diva(C, E, F);
					scamp5_kernel_end();
				}
				val >>= 1;
			}

			// store back the value of LPF
			scamp5_kernel_begin();
				WHERE(DREG_PE);
					bus(NEWS, B);

				WHERE(DREG_STO2);
					bus(BIPOLAR_LPF_REGISTER_STO2, XN);
			scamp5_kernel_end();

			if( compute_blur_for_approximation)
			{
				scamp5_kernel_begin();
					WHERE(DREG_PE);
						mov(F, B);
						move_the_image_for_F();

					SET(R1, R2);
					gauss(F, F, BLUR_ITERATIONS_BIPOLAR);
				scamp5_kernel_end();


				scamp5_kernel_begin();
					mov(B, F);
				scamp5_kernel_end();
			}

			// by now
			// A : I_OPL
			// B : G_A * E_A * Q(V_bip)
			// C : garbage
			// D: V_bip

			// compute the 3 MSBs of B
			// if B > 64 ?
			// add half of bipolar cell to what will be subtracted from

			// "A/D conversion"

			scamp5_load_in(F, 64);
			scamp5_kernel_begin();
				WHERE(DREG_PE);
					sub(C, B, F);
					where(C);
				// FLAG set to C > 64
				MOV(VBIP_GREATER_THAN_64 , FLAG);
				AND(VBIP_GREATER_THAN_64 , VBIP_GREATER_THAN_64 , DREG_PE);

				WHERE(VBIP_GREATER_THAN_64 );
					sub(B, B, F);
			scamp5_kernel_end();

			scamp5_load_in(F, 32);
			scamp5_kernel_begin();
				WHERE(DREG_PE);
					sub(C, B, F);
					where(C);
				// FLAG set to C > 32
				MOV(VBIP_GREATER_THAN_32 , FLAG);
				AND(VBIP_GREATER_THAN_32 , VBIP_GREATER_THAN_32, DREG_PE);

				WHERE(VBIP_GREATER_THAN_32);
					sub(B, B, F);
			scamp5_kernel_end();

			scamp5_load_in(F, 16);
			scamp5_kernel_begin();
				WHERE(DREG_PE);
					sub(C, B, F);
					where(C);
				// FLAG set to C > 32
				MOV(VBIP_GREATER_THAN_16 , FLAG);
				AND(VBIP_GREATER_THAN_16, VBIP_GREATER_THAN_16, DREG_PE);

				WHERE(VBIP_GREATER_THAN_16);
					sub(B, B, F);
			scamp5_kernel_end();

			// By this point
			// A: I_opl
			// B: garbage ( least significant bits of G_A * E_A * Q(V_bip) )
			// C: garbage
			// D: V_bip
			// S3: VBIP_GREATER_THAN_64
			// S4: VBIP_GREATER_THAN_32
			// S5: VBIP_GREATER_THAN_16

			// is now time to multiply V_bip by 0b{S3,S4,S5}

			if (add_leftovers_to_feedback)
			{
				// use the remainder of B ( it should be V_Bip / 8)

				scamp5_kernel_begin();
					WHERE(DREG_PE);
						// move the V_bip to C
						mov(C, D);

						// V_Bip / 2 in C
						diva(C, E, F);

					WHERE(VBIP_GREATER_THAN_64);
						add(B, B, C);

					WHERE(DREG_PE);
						// V_Bip / 4 in C
						diva(C, E, F);

					WHERE(VBIP_GREATER_THAN_32);
						add(B, B, C);

					WHERE(DREG_PE);
						// V_Bip / 8 in C
						diva(C, E, F);

					WHERE(VBIP_GREATER_THAN_16);
						add(B, B, C);

				scamp5_kernel_end();

			}
			else
			{
				// zero out the register B
				scamp5_load_in(F, 0);
				scamp5_kernel_begin();
					WHERE(DREG_PE);
						mov(B, F);

						// move the V_bip to C
						mov(C, D);

						// V_Bip / 2 in C
						diva(C, E, F);

					WHERE(VBIP_GREATER_THAN_64);
						add(B, B, C);

					WHERE(DREG_PE);
						// V_Bip / 4 in C
						diva(C, E, F);

					WHERE(VBIP_GREATER_THAN_32);
						add(B, B, C);

					WHERE(DREG_PE);
						// V_Bip / 8 in C
						diva(C, E, F);

					WHERE(VBIP_GREATER_THAN_16);
						add(B, B, C);

				scamp5_kernel_end();
			}
		}

		// By this point the difference between the I_OPL and the alpha_ad * V_bip can be computed
		// by simply A := A - B
		if (compute_feedback_phase2)
		{
			scamp5_kernel_begin();
				WHERE(DREG_PE);
					sub(A, A, B);
			scamp5_kernel_end();

			scamp5_load_in(F, 0);
			scamp5_kernel_begin();
				WHERE(DREG_PE);
					mov(B, F);
			scamp5_kernel_end();

			int val = 1 << (alpha_fixed.no_bits - 1);

			while (val)
			{
				if (val & alpha_fixed.value)
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

				val >>= 1;
			}

			// the output from stage 2 is the sum of the bipolar cells (currently stored in D) and the value comptued in A
			scamp5_kernel_begin();
				WHERE(DREG_PE);
					add(A, A, D);


				// store back the bipolar cells
					bus(NEWS, A);

				WHERE(DREG_STO1);
					bus(BIPOLAR_REGISTER_STO1, XW);
			scamp5_kernel_end();
		}





		fill_A();


		first_pass = false;

		if(vs_gui_is_on()){
			scamp5_output_image(A, display_00);
			scamp5_output_image(F, display_01);
			//scamp5_output_image(S2, display_10);
			//scamp5_output_image(S3, display_11);

            vs_post_text("(%lu) %s\n",VS_LC,scamp5_verbose_str());
		}
    }

	return 0;
}
