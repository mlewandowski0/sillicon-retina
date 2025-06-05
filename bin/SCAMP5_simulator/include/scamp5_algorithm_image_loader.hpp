/*************************************************************************
 * SCAMP Vision Chip Development System Library
 *------------------------------------------------------------------------
 * Copyright (c) 2020 The University of Manchester. All Rights Reserved.
 *
 *************************************************************************/

/*!

\file

\ingroup VS_SCAMP5_MODULE

\author Jianing Chen

*/

#ifndef INC_SCAMP5_ALGORITHM_IMAGE_LOADER_HPP_
#define INC_SCAMP5_ALGORITHM_IMAGE_LOADER_HPP_


#include <scamp5_algorithm.hpp>


namespace scamp5_algorithm{

/*!
A class for loading images through the Host App.
*/
class image_loader{

protected:
	const char*filepath_format;
	uint16_t image_index_first;
	uint16_t image_index_last;
	bool do_reverse;
	vs_handle progress_bar;
	int16_t playback_state;
	int16_t playback_progress;
	int16_t verbose_mode;
	uint16_t image_index;
	uint16_t image_width;
	uint16_t image_height;
	scamp5_kernel kernel_1;
	scamp5_kernel kernel_2;
	std::function<void(int8_t)> _areg_dst_in;
	std::function<void(int8_t)> _areg_tmp_in;
#ifdef VS_SIM_SOURCE
	std::function<void(const char*,uint8_t)> _sim_load_areg;
	std::function<void(const char*,uint8_t)> _sim_load_dreg;
#else
	std::function<void(vs_dotmat const&,uint8_t,uint8_t)> _dreg_dst_load;
	std::function<void(vs_dotmat const&,uint8_t,uint8_t)> _dreg_tmp_load;
#endif

	void _load_video_frame(int idx_inc,int type,bool centering);

public:
	static const int D2A_BITS = 5;

    /*!
    	@brief get the index of the image when in frame sequence mode
    */
	inline int get_image_index(){
		return image_index;
	};

    /*!
    	@brief get the width of the previous loaded image
    */
	inline uint16_t get_image_width(){
		return image_width;
	};

    /*!
    	@brief get the height of the previous loaded image
    */
	inline uint16_t get_image_height(){
		return image_height;
	};

    /*!
    	@brief set verbose level
    */
	inline void set_verbose_mode(int16_t v){
		verbose_mode = v;
	}

    /*!
    	@brief construct the algorithm class
    	@param _dst1 	destination AREG when loading an analog image
    	@param _dst2 	destination DREG when loading an digital image
    	@param _tmp1 	a temporary AREG needed by the internal operations
    	@param _tmp2 	a temporary DREG needed by the internal operations
    */
	image_loader(AREG _dst1,DREG _dst2,AREG _tmp1,DREG _tmp2);

	std::function<void(void)> waiting_callback;

    /*!
    	@brief add a slider to the host GUI to show video progress
    */
	void add_gui_items();

    /*!
    	@brief load a image file as a greyscale image
    */
	void load_areg_image(const char*filepath,bool centering=true,bool halfscale=false);

    /*!
    	@brief load a image file as a Boolean image
    */
	void load_dreg_image(const char*filepath,bool centering=true);

    /*!
    	@brief initialize frame sequence loading mode

    	@param filepath		file path string compatible with printf-like functions
    	@param first_idx 	index number of the first frame
    	@param last_idx		index number of the last frame
    	@param reverse		whether to play backwards when end

    	Note, for real devices \p filepath is relative to the executable file of the Host App connected.
    	For simulation, \p filepath is relative to the simulation server executable.

    	In addition, \p filepath should contain frame index number field that is compatible with printf-like functions.
    	For example, the following code setup a continuous image loader from "frame_0200.bmp"
    	to "frame_0800.bmp" under "D:/Dataset/ScampInput/":
    	\code
    	static scamp5_algorithm::image_loader image_loader_C_R9(C,R9,D,R10);
		const char*video_filepath = "D:/Dataset/ScampInput/frame_%04d.bmp";
		int first_idx = 200;
		int last_idx = 800;
		image_loader_C_R9.init_video_frames(video_filepath, first_idx, last_idx, false);
    	\endcode

    */
	void init_video_frames(const char*filepath,uint16_t first_idx,uint16_t last_idx,bool reverse=false);

    /*!
    	@brief load the next frame in the frame sequence as a DREG image
    */
	inline void load_video_frame_dreg(int idx_inc=1,bool centering=true){
		return _load_video_frame(idx_inc,0,centering);
	}

    /*!
    	@brief load the next frame in the frame sequence as a AREG image
    */
	inline void load_video_frame_areg(int idx_inc=1,bool centering=true){
		return _load_video_frame(idx_inc,1,centering);
	}

    /*!
    	@brief load the next frame in the frame sequence as a AREG halfscale image
    */
	inline void load_video_frame_areg_halfscale(int idx_inc=1,bool centering=true){
		return _load_video_frame(idx_inc,2,centering);
	}
};


};

#endif
