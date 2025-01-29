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

#ifndef INC_SCAMP5_ALGORITHM_MORPHOLOGY_HPP_
#define INC_SCAMP5_ALGORITHM_MORPHOLOGY_HPP_


#include <scamp5_algorithm.hpp>


namespace scamp5_algorithm{


class dnews_kernels{
protected:
	static scamp5_kernel kernel_dnews_all;
	static scamp5_kernel kernel_dnews_ew;
	static scamp5_kernel kernel_dnews_sn;
	dnews_kernels();
};


/*!
A class for commonly used binary morphology functions.
*/
class morphology:protected dnews_kernels{

protected:
	scamp5_kernel kernel_init;
	scamp5_kernel kernel_erode;
	scamp5_kernel kernel_dilate;
	scamp5_kernel kernel_shrink;
	scamp5_kernel kernel_edge;

public:

    /*!
    	@brief construct the algorithm class
    	@param _dst 	the DREG to store the result
    	@param _src 	the DREG of the input image
    	@param _tmp1 	a temporary DREG to hold intermediate results
    	@param _tmp2 	a temporary DREG to hold intermediate results
    	@param _tmp3 	a temporary DREG to hold intermediate results
    */
	morphology(DREG _dst, DREG _src, DREG _tmp1, DREG _tmp2, DREG _tmp3);

    /*!
    	@brief run erosion given the number of iterations
    */
	void erosion(int iterations,int mode=0);

    /*!
    	@brief run dilation given the number of iterations
    */
	void dilation(int iterations,int mode=0);

    /*!
    	@brief run digital edge detection
    */
	void edge_detection();

    /*!
    	@brief run shrink given the number of iterations

    	"shrink" is a process similar to erosion, but it will not eliminate the last pixel of a blob.
    */
	void shrink(int iterations);

};


};

#endif /* INC_SCAMP5_ALGORITHM_MORPHOLOGY_HPP_ */
