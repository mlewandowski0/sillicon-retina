/*************************************************************************
 * SCAMP Vision Chip Development System Library
 *------------------------------------------------------------------------
 * Copyright (c) 2020 The University of Manchester. All Rights Reserved.
 *
 *************************************************************************/

/*!

\file 

\ingroup VS_M0_CORE_MODULE

\author Jianing Chen

*/

#ifndef VS_INTERACTION_REQUEST_HPP
#define VS_INTERACTION_REQUEST_HPP

#include <array>
#include <functional>
#include <vs_common.h>
#include <vs_protocol_gui.hpp>


/*!

    @brief request a file from the host computer

*/
int vs_gui_request_file(const char*filepath,std::function<void(const uint8_t*chunk,size_t bytes,size_t bytes_remain)> cb,uint32_t timeout_msec=1000);


/*!

    @brief request a binary image from the host computer

    @param filepath    	filepath relative to the host GUI
    @param bits 		bits per pixel (highest bit first)
    @param cb 			callback function
    @return 			none zero if there is already a request undergoing

	The callback function will be called when the image is received or when a timeout occurs.
	It is necessary to keep frame loop running (or execute ::vs_process_message) to keep the polling system working on the M0 core.
	If there is nothing else to do until the image arrive, a waiting loop could be used. For example:
	\code
	vs_gui_request_image("test.bmp",1,[&](vs_dotmat const& dotmat,int s){
		if(s<0){
			vs_post_text("image not received in time!\n");
			return;
		}
		// ...
	});
	do{
		vs_process_message();
	}while(!vs_gui_request_done());
	\endcode
*/
int vs_gui_request_image(const char*filepath,int bits,std::function<void(vs_dotmat const&,int)> cb,uint32_t timeout_msec=1000);


/*!
    @brief get whether there is a ongoing request
*/
bool vs_gui_request_done();


#endif
