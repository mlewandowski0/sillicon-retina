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

#ifndef INC_SCAMP5_ALGORITHM_EDGE_DETECTION_HPP_
#define INC_SCAMP5_ALGORITHM_EDGE_DETECTION_HPP_


#include <scamp5_algorithm.hpp>


namespace scamp5_algorithm{

/*!
A class for edge detection functions on AREG images.
*/
class edge_detection{

protected:
	scamp5_kernel kernel_1;
	scamp5_kernel kernel_2;
	scamp5_kernel kernel_3;
	std::function<void(int8_t)> _dst_in;

	enum edge_type{gussian,sobel,robert};
	void run(edge_type type_enum);

public:

    /*!
    	@brief construct the algorithm class
    	@param _dst 	the AREG to store the result
    	@param _src 	the AREG of the input image
    	@param _tmp1 	a temporary AREG to hold intermediate results
    	@param _tmp2 	a temporary AREG to hold intermediate results

    	Note: the sobel edge detection in this class will modify the '_src' AREG
    	as it will turn the input image to a half-scale image.
    */
	edge_detection(AREG _dst, AREG _src, AREG _tmp1, AREG _tmp2);

    /*!
    	@brief run Difference of Gaussian
    */
	inline void gussian_edge(){
		run(gussian);
	};

    /*!
    	@brief run Sobel Edge Detection
    */
	inline void sobel_edge(){
		run(sobel);
	};

    /*!
    	@brief run Robert Edge Detection
    */
	inline void robert_edge(){
		run(robert);
	};
};



};

#endif
