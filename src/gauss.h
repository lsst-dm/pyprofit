#ifndef __PYPROFIT_GAUSS_H_
#define __PYPROFIT_GAUSS_H_

#ifndef PROFIT_PROFIT_H
#include "profit/profit.h"
#endif

profit::Image profit_make_gaussian(
    const double XCEN, const double YCEN, const double MAG, const double RE,
    const double ANG, const double AXRAT,
    const double XMIN, const double XMAX, const double YMIN, const double YMAX,
    const unsigned int XDIM, const unsigned int YDIM,
    const double ACC);

#endif