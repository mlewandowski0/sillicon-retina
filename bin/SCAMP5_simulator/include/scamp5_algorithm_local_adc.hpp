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

#ifndef INC_SCAMP5_ALGORITHM_LOCAL_ADC_HPP_
#define INC_SCAMP5_ALGORITHM_LOCAL_ADC_HPP_


#include <scamp5_algorithm.hpp>


namespace scamp5_algorithm{

/*!
A class for convert an AREG image to 5 DREG images (5-bit/32 greyscale level) locally on the PE using successive approximation algorithm.
This method works faster than ::scamp5_algorithm::local_adc_ramp, but only works for -128 to 127 full range image.

The utility function scamp5_algorithm::output_image_5bit can be used to display the converted result
as a single greyscale image (rather than 5 DERG image) on the host app.
*/
class local_adc_bin{

protected:
	std::function<void(int8_t)> _tmp_in;
	scamp5_kernel init;
	scamp5_kernel areg_flag;
	scamp5_icw mov_to_bit[5];
	scamp5_kernel sub_tmp2;

public:
    /*!
    	@brief construct the algorithm class

    	This algorithm does not modify \p _src.
    */
	local_adc_bin(DREG _bit4, DREG _bit3, DREG _bit2, DREG _bit1, DREG _bit0, AREG _src, AREG _tmp, AREG _tmp2);

    /*!
    	@brief do the 5-bit A2D conversion
    */
	void convert_areg_to_5bits();

};


/*!
A class for convert an AREG image to 4 DREG images (4-bit/16 greyscale level) locally on the PE using ramp algorithm.
This method works slower than ::scamp5_algorithm::local_adc_bin, but allow for a smaller grey-scale band to be converted.

The utility function scamp5_algorithm::output_image_4bit can be used to display the converted result
as a single greyscale image (rather than 4 DERG images) on the host app.
*/
class local_adc_ramp{

protected:
	std::function<void(int8_t)> _tmp_in;
	scamp5_kernel clr_all_bits;
	scamp5_kernel bit0_write0;
	scamp5_kernel bit0_write1;
	scamp5_kernel bit1_write0;
	scamp5_kernel bit1_write1;
	scamp5_kernel bit2_write0;
	scamp5_kernel bit2_write1;
	scamp5_kernel bit3_write0;
	scamp5_kernel bit3_write1;
	scamp5_kernel areg_ramp_cmp;

	void _write_4bits_flagged(uint8_t b);

public:
    /*!
    	@brief construct the algorithm class

    	This algorithm does not modify \p _src.
    */
	local_adc_ramp(DREG _bit3, DREG _bit2, DREG _bit1, DREG _bit0, AREG _src, AREG _tmp);

    /*!
    	@brief do the 4-bit A2D conversion
    	@param floor 	AREG value which will be 0 in the conversion
    	@param step 	what each of the 16 levels equals to in AREG value
    */
	void convert_areg_to_4bits(int8_t floor=-128,int8_t step=16);

};



/*!
	@brief display a 4-bit greyscale image using 4 DREG image

	It is twice as fast to convert a AREG image to 4-bit image then output using this function than
	send the AREG image directly.
*/
inline void output_image_4bit(vs_handle display,DREG _bit3, DREG _bit2, DREG _bit1, DREG _bit0){
    scamp5_output_bitstack_begin(display,4);
    scamp5_output_bitstack_bit(_bit3);
    scamp5_output_bitstack_bit(_bit2);
    scamp5_output_bitstack_bit(_bit1);
    scamp5_output_bitstack_bit(_bit0);
    scamp5_output_bitstack_end();
}


/*!
	@brief display a 5-bit greyscale image using 5 DREG image

	It is faster to convert a AREG image to 5-bit image then output using this function than
	send the AREG image directly.
*/
inline void output_image_5bit(vs_handle display,DREG _bit4, DREG _bit3, DREG _bit2, DREG _bit1, DREG _bit0){
    scamp5_output_bitstack_begin(display,5);
    scamp5_output_bitstack_bit(_bit4);
    scamp5_output_bitstack_bit(_bit3);
    scamp5_output_bitstack_bit(_bit2);
    scamp5_output_bitstack_bit(_bit1);
    scamp5_output_bitstack_bit(_bit0);
    scamp5_output_bitstack_end();
}


};
#endif /* INC_SCAMP5_ALGORITHM_LOCAL_ADC_HPP_ */
