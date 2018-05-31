#ifndef __PYPROFIT_PYPROFIT_H_
#define __PYPROFIT_PYPROFIT_H_

#include <Python.h>

/* Exceptions */
static PyObject *profit_error;

/* Macros */
#define PYPROFIT_RAISE(str) \
	do { \
		PyErr_SetString(profit_error, str); \
		return NULL; \
	} while (0)

#endif