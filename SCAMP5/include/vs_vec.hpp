/*************************************************************************
 * SCAMP Vision Chip Development System Library
 *------------------------------------------------------------------------
 * Copyright (c) 2020 The University of Manchester. All Rights Reserved.
 *
 *************************************************************************/

#ifndef VS_VEC_HPP
#define VS_VEC_HPP

#include <cstdint>
#include <array>


struct vs_vec_int8_t{
	int8_t x;
	int8_t y;
	inline int8_t* data(){
		return (int8_t*)this;
	}
};


struct vs_vec_int16_t{
	int16_t x;
	int16_t y;
	inline int16_t* data(){
		return (int16_t*)this;
	}
};


struct vs_vec_int32_t{
	int32_t x;
	int32_t y;
	inline int32_t* data(){
		return (int32_t*)this;
	}
};


struct vs_vec_xy_t{
	uint8_t x;
	uint8_t y;
	inline uint8_t* data(){
		return (uint8_t*)this;
	}
};


struct vs_vec_rc_t{
	uint8_t row;
	uint8_t col;
	inline uint8_t* data(){
		return (uint8_t*)this;
	}
};


struct vs_vec_aabb_t{
	uint8_t row0;
	uint8_t col0;
	uint8_t row1;
	uint8_t col1;
	inline uint8_t* data(){
		return (uint8_t*)this;
	}
};


class vs_vec_roi{

public:
	union{
		struct{
			int16_t y0;
			int16_t x0;
			int16_t y1;
			int16_t x1;
		};
		int16_t data[4];
	};

	inline void convert_from(const uint8_t*rect4v){
		y0 = rect4v[0];
		x0 = rect4v[1];
		y1 = rect4v[2];
		x1 = rect4v[3];
	};

	inline void convert_from(vs_vec_aabb_t const&aabb){
		y0 = aabb.row0;
		x0 = aabb.col0;
		y1 = aabb.row1;
		x1 = aabb.col1;
	};

	vs_vec_roi(){
		x0 = 0;
		y0 = 0;
		x1 = 0;
		y1 = 0;
	};

	vs_vec_roi(const uint8_t*rect4v){
		convert_from(rect4v);
	};

	vs_vec_roi(vs_vec_aabb_t const&aabb){
		convert_from(aabb);
	};

	vs_vec_roi(int16_t centroid_x,int16_t centroid_y,int16_t width,int16_t height){
		x0 = centroid_x - width/2;
		x1 = centroid_x + (width + 1)/2 - 1;
		y0 = centroid_y - height/2;
		y1 = centroid_y + (height + 1)/2 - 1;
	};

	inline void convert_to(uint8_t*rect4v)const{
		rect4v[0] = y0;
		rect4v[1] = x0;
		rect4v[2] = y1;
		rect4v[3] = x1;
	};

	inline void clamp(){
		for(int i=0;i<4;i++){
			data[i] = std::max<int16_t>(0,data[i]);
			data[i] = std::min<int16_t>(255,data[i]);
		}
	};

	inline void bound(){
		if(x0<0){
			x1 -= x0;
			x0 = 0;
		}
		if(y0<0){
			y1 -= y0;
			y0 = 0;
		}
		if(x1>255){
			x0 -= (x1 - 255);
			x1 = 255;
		}
		if(y1>255){
			y0 -= (y1 - 255);
			y1 = 255;
		}
	};

	inline void move(int16_t dx,int16_t dy){
		x0 += dx;
		y0 += dy;
		x1 += dx;
		y1 += dy;
	};

	inline int get_width()const{
		return x1 - x0 + 1;
	};

	inline int get_height()const{
		return y1 - y0 + 1;
	};

	inline int get_centroid_x()const{
		return (x0 + x1 + 1)/2;
	};

	inline int get_centroid_y()const{
		return (y0 + y1 + 1)/2;
	};

	inline void resize(int16_t width,int16_t height){
		auto centroid_x = get_centroid_x();
		auto centroid_y = get_centroid_y();
		x0 = centroid_x - width/2;
		x1 = centroid_x + (width + 1)/2 - 1;
		y0 = centroid_y - height/2;
		y1 = centroid_y + (height + 1)/2 - 1;
	};

};


#endif /* VS_VEC_HPP */
