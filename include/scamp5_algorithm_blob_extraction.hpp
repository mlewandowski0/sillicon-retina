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

#ifndef INC_SCAMP5_ALGORITHM_BLOB_EXTRACTION_HPP_
#define INC_SCAMP5_ALGORITHM_BLOB_EXTRACTION_HPP_


#include <scamp5_algorithm.hpp>


namespace scamp5_algorithm{

/*!
A class for blob extraction algorithm.
*/
class blob_extraction{

protected:
	std::function<void(uint8_t*,uint16_t)> scan_events;
	std::function<void(uint8_t,uint8_t)> load_point;
	std::function<void(uint8_t*)> scan_boundingbox;
	std::function<void(void)> flooding;
	scamp5_kernel kernel_1;
	scamp5_kernel kernel_2;

public:

    /*!
    	@brief construct the algorithm class
    	@param _src 	the source DREG to scan the blob from
    	@param _tmp1 	a temporary DREG to hold intermediate results
    	@param _tmp2 	a temporary DREG to hold intermediate results
    	@param flood_steps 	number of flooding step to use in the algorithm

    	Note: This algorithm class also need RN RE RW RS RF R12 as the temporary registers.

    	The default value of 'flood_steps' works in most situations.
    	A bigger 'flood_steps' may be needed to extract large blobs, but it will make the algorithm run slower.
    */
	blob_extraction(DREG _src, DREG _tmp1, DREG _tmp2, int flood_steps=8);

    /*!
    	@brief run the algorithm
    	@param aabb_buffer 	the bounding box coordinates output buffer
    	@param max_num 		maximum number of bounding box to scan
    	@param count 		number of blob extracted will be given via this pointer
    */
	void run(uint8_t*aabb_buffer,size_t max_num,size_t*count);

};


};

#endif
