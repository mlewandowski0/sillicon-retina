/*************************************************************************
 * SCAMP Vision Chip Development System Library
 *------------------------------------------------------------------------
 * Copyright (c) 2020 The University of Manchester. All Rights Reserved.
 *
 *************************************************************************/

#ifndef VS_MODEL_H
#define VS_MODEL_H

#define VS_MODEL	501

#if VS_MODEL/100==5
#define VS_MODEL_SCAMP5
#elif VS_MODEL/100==7
#define VS_MODEL_SCAMP7
#endif

#endif
