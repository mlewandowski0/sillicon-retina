/*************************************************************************
 * SCAMP Vision Chip Development System Library
 *------------------------------------------------------------------------
 * Copyright (c) 2020 The University of Manchester. All Rights Reserved.
 *
 *************************************************************************/
/*
* Scamp5d M0 Example - Simulation
*
* Note: this source code works on both of the hardware and the simulation 
*       but with an emphasis on demonstrating simulation-related functions.
*/

#define filter_size 4
#define filter_size_sq (filter_size*filter_size)

#include <math.h>
#include <scamp5.hpp>
#include <list>
#include <string>

#include "TIMING_STATS.hpp"

using namespace SCAMP5_PE;
using namespace std;

void MAIN_MNIST_SINGLE_LAYER_16();
void great_sad();

using namespace SCAMP5_PE;

volatile int threshold;

int main(){

    vs_sim::config("server_ip","127.0.0.1");
    vs_sim::config("server_port","27715");
    
    // Initialization
    vs_init();

    vs_sim::enable_keyboard_control();// this allow a few shortcuts to be used. e.g. 'Q' to quit. 
	vs_sim::reset_model(3);// reset model is also used to configure the error model


    // Setup Host GUI
	vs_gui_set_info(VS_M0_PROJECT_INFO_STRING);
    

    vs_on_gui_update(VS_GUI_FRAME_RATE, [&](int32_t new_value) {
        uint32_t framerate = new_value;
        if (framerate > 0) {
            vs_frame_trigger_set(1, framerate);
            vs_enable_frame_trigger();
            vs_post_text("frame trigger: 1/%d\n", (int)framerate);
        }
        else {
            vs_disable_frame_trigger();
            vs_post_text("frame trigger disabled\n");
        }
        });

    vs_on_host_connect([&]() {
        vs_post_text("Scamp5d Image Storage\n");
        vs_post_text("loop_counter: %d\n", (int)vs_loop_counter_get());
        scamp5_kernel::print_debug_info();
        vs_led_on(VS_LED_2);
        });

    vs_on_host_disconnect([&]() {
        vs_led_off(VS_LED_2);
        });

    auto slider_vxc = vs_gui_add_slider("vxc: ", 0, 4095, 2900);
    vs_on_gui_update(slider_vxc, [&](int32_t new_value) {
        vs_scamp5_configure_voltage(3, new_value);
        });

    MAIN_MNIST_SINGLE_LAYER_16();



    return 0;
}