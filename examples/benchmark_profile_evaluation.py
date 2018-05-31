#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#

import argparse
import collections
import itertools
import math
import sys
import time
import timeit

import galsim
import numpy as np
import pandas as pd
import pyprofit

def powers_of_to_up_to(n):
    # Surely it's easier than this, but whatever...
    last_exponent = int(math.floor(math.log(n, 2)))
    powers = set([2**x for x in range(1, last_exponent + 1)] + [n])
    powers = list(powers)
    powers.sort()
    if 1 in powers:
        powers.remove(1)
    return powers

parser = argparse.ArgumentParser('')
parser.add_argument('-n', '--niter', help='Number of iterations, defaults to 100',
                    type=int, default=100)
parser.add_argument('-w', '--width', help='Image width',
                    type=int, default=200)
parser.add_argument('-H', '--height', help='Image height',
                    type=int, default=200)
parser.add_argument('-t', '--omp_threads', help='Maximum OpenMP threads to use for profile evaluation, defaults to 1',
                    type=int, default=1)
parser.add_argument('-N', '--nsers', help='Sersic indexes to sample, defaults to 1,10,10', default=None)
parser.add_argument('-a', '--angs', help='Angles to sample, defaults to 0,45,6', default=None)
parser.add_argument('-A', '--axrats', help='Axis ratios to sample, defaults to 0.1,1,4', default=None)
parser.add_argument('-r', '--res', help='Re values sample, defaults to 1,width/2,5', default=None)
parser.add_argument('-b', '--boxes', help='Boxiness values to sample, defaults to -0.5,0.5,3', default=None)
parser.add_argument('-o', '--opencl', help='Benchmark libprofit OpenCL', default=False, type=bool)
parser.add_argument('-g', '--galsim', help='Benchmark GalSim integration', default=False, type=bool)
parser.add_argument('-gfft', '--galsimfft', help='Benchmark GalSim Fourier integration', default=False, type=bool)
parser.add_argument('-psf', '--psffwhm', help='Gaussian PSF FWHM', default=0, type=float)
parser.add_argument('-s', '--oversample', help='libprofit oversampling factor', default=1, type=int)
parser.add_argument('-p', '--path', help='Output path for results', default=None, type=str)

args = parser.parse_args()

n_iter = args.niter
width = args.width
height = args.height
omp_threads = powers_of_to_up_to(args.omp_threads)
fileout = args.path
if fileout is not None:
    fileout += '{:d}x{:d}_psffwhm-{:.3e}_oversample-{:d}.dat'.format(width, height, args.psffwhm, args.oversample)
    fileout = open(fileout, 'w')


def define_parameter_range(name, spec):
    spec_elements = list(map(float, spec.split(',')))
    if len(spec_elements) in (1, 2):
        return spec_elements
    Min, Max, n = spec_elements
    diff = Max - Min
    step = diff/(n - 1)
    values = np.arange(Min, Max + step, step)

    # Avoid rounding errors
    if values[-1] != Max:
        values[-1] = Max

    print("%d %s: %r" % (n, name, values))
    return values


nsers = define_parameter_range('nser', args.nsers or '1,10,10')
angs = define_parameter_range('angs', args.angs or '0,45,6')
axrats = define_parameter_range('axrats', args.axrats or '0.1,1,4')
res = define_parameter_range('res', args.res or '1,%f,5' % (width/2.,))
boxes = define_parameter_range('boxes', args.boxes or '-0.5,0.5,3')

print("Benchmark measuring profile image of %d x %d with %d iterations" % (width, height, n_iter,))
print("\n%d combinations to be benchmarked" % (len(nsers) * len(angs) * len(axrats) * len(res) * len(boxes)))
print("Parameter ranges: ")


# Patch timeit to return rval as well as per:
# https://stackoverflow.com/questions/24812253/how-can-i-capture-return-value-with-python-timeit-module
timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""

# What we use to time iterative executions
timing_result = collections.namedtuple("TimingResult", ['time', 'errorabs', 'negsum', 'exception'])


def time_pyprofit(imgref = None, **kwargs):
    try:
        def pyprofit_wrap():
            return pyprofit.make_model(kwargs)

        acc = 0
        timerun, rv = timeit.timeit(pyprofit_wrap, number=1, setup="import pyprofit")
        for i in range(n_iter-1):
            newtime, _ = timeit.timeit(pyprofit_wrap, number=1, setup="import pyprofit")
            timerun = np.min([timerun, newtime])

        rv = np.array(np.matrix(rv[0]))
        if kwargs["oversample"] > 1:
            from skimage.measure import block_reduce
            rv = block_reduce(rv, (kwargs["oversample"],kwargs["oversample"]))
            rv = rv[0:rv.shape[0], 0:rv.shape[1]]
        if imgref is not None:
            acc = np.sum(np.abs(rv - imgref))
        negsum = np.sum(rv[rv<0])
        return timing_result(timerun, acc, negsum, None)
    except pyprofit.error as error:
        return timing_result(0, 0, 0, error)


def time_galsim(object, imgref = None, **kwargs):
    try:
        def galsim_wrap():
            return object.drawImage(**kwargs)

        acc = 0
        timerun, rv = timeit.timeit(galsim_wrap, number=n_iter, setup="import galsim")
        rv = rv.array
        if imgref is not None:
            acc = np.sum(np.abs(rv - imgref))
        negsum = np.sum(rv[rv < 0])
        return timing_result(timerun / n_iter, acc, negsum, None)
    except Exception as error:
        return timing_result(0, 0, 0, error)


if args.psffwhm > 0:
    widthpsf = math.ceil(width/2.)*args.oversample
    heightpsf = math.ceil(height/2.) * args.oversample
    widthpsf += 1 - (widthpsf % 2)
    heightpsf += 1 - (heightpsf % 2)
    angpsf = 0
    axratpsf = 1
    imgpsf = galsim.ImageD(widthpsf, heightpsf, scale=1)
    objpsf = galsim.Sersic(n=0.5, half_light_radius=args.psffwhm*args.oversample/2.0, flux=1).shear(
        g=(1 - axratpsf) / (1 + axratpsf), beta=angpsf * galsim.degrees)
    psfimg = objpsf.drawImage(imgpsf, method="real_space").array
    conv = pyprofit.make_convolver(width=width*args.oversample, height=height*args.oversample, psf=psfimg,
                                   convolver_type='fft', reuse_psf_fft=True)
    # Reset for galsim, which doesn't need explicit oversampling
    objpsf = galsim.Sersic(n=0.5, half_light_radius=args.psffwhm / 2.0, flux=1)

openclenvs = []

if args.opencl:
    # Get the OpenCL platforms/devices information
    cl_info = pyprofit.opencl_info()
    def all_cl_devs():
        return ((p, d, cl_info[p][2][d][1]) for p in range(len(cl_info)) for d in range(len(cl_info[p][2])))

    # Print OpenCL information onto the screen
    print("\nOpenCL platforms/devices information:")
    for plat, dev, has_double_support in all_cl_devs():
        print("[%s] %s / %s. Double: %s" % (
            '%d%d' % (plat, dev),
            cl_info[plat][0], cl_info[plat][2][dev][0],
            "Yes" if cl_info[plat][2][dev][1] else "No"))
    print('')

    # Get an float OpenCL environment for each of them
    # If the device supports double, get a double OpenCL environment as well
    sys.stdout.write('Getting an OpenCL environment for each of them now...')
    sys.stdout.flush()
    for p, dev, double_support in all_cl_devs():
        openclenvs.append(pyprofit.openclenv(p, dev, False))
        if double_support:
            openclenvs.append(pyprofit.openclenv(p, dev, True))
    print(' done!')

sersic_profile = {'xcen': width*args.oversample/2, 'ycen': height*args.oversample/2, 'mag': 0, 'rough': 0, 'convolve': args.psffwhm > 0}
profiles = {'sersic': [sersic_profile]}

# Evaluate with an empty OpenCL environment (i.e., use CPU evaluation),
# then each of the OpenCL environments in turn, and then with different
# OpenMP threads
labels = ['CPU']
if args.opencl:
    for p, dev, double_support in all_cl_devs():
        labels.append('CL_%d%d_f' % (p, dev))
        if double_support:
            labels.append('CL_%d%d_d' % (p, dev))
labels += ['OMP_%d' % t for t in omp_threads]

eval_args = [{}]
eval_args += [{'openclenv': clenv} for clenv in openclenvs]
eval_args += [{'omp_threads': t} for t in omp_threads]

methods_galsim = []
methods_galsim_short = {
    "real_space": "rs"
    , "fft": "ft"
}
labels_galsim = {}
labels_galsim_short = {}
if args.galsim:
    if not args.psffwhm > 0:
        methods_galsim.append("real_space")
    if args.galsimfft:
        methods_galsim.append("fft")

for method in methods_galsim:
    labels_galsim[method] = "CPU_GS_" + method
    labels_galsim_short[method] = "CPU_GS_" + methods_galsim_short[method]

parameters = (nsers, angs, axrats, res, boxes)
times = collections.defaultdict(list)
for label_and_evalargs in zip(labels, eval_args):
    label, evalargs = label_and_evalargs
    if args.psffwhm > 0:
        evalargs.update({"convolver": conv, "psf": psfimg})
    evalargs.update({"oversample": args.oversample})

    sys.stdout.write('Evaluating %s...\n' % label)
    sys.stdout.flush()

    start = time.time()
    for nser, ang, axrat, re, box in itertools.product(*parameters):
        sersic_profile['nser'] = nser
        sersic_profile['ang'] = ang-90
        sersic_profile['axrat'] = axrat
        sersic_profile['re'] = re*args.oversample
        sersic_profile['box'] = box
        imgrefarr = None
        if args.galsim:
            imgref = galsim.ImageD(width, height, scale=1)
            objref = galsim.Sersic(
                n=nser, half_light_radius=re * np.sqrt(axrat), flux=1,
                gsparams=galsim.GSParams(maximum_fft_size=12 * 4096, realspace_relerr=1e-6,
                realspace_abserr=1e-10)).shear(g=(1 - axrat) / (1 + axrat), beta=ang * galsim.degrees)
            if args.psffwhm > 0:
                if nser == 0.5:
                    angs = np.array([angpsf, ang])
                    fwhms = np.array([args.psffwhm, 2*re])
                    costh = np.cos(angs*np.pi / 180)
                    sinth = np.sin(angs*np.pi / 180)

                    fwhmconv = {"x": np.sum(np.square(fwhms) * costh * np.abs(costh)),
                                "y": np.sum(np.square(fwhms) * sinth * np.abs(sinth))}
                    for key, value in fwhmconv.items():
                        fwhmconv[key] = np.sign(value) * np.sqrt(np.abs(value))
                    angconv = np.arctan2(fwhmconv["y"], fwhmconv["x"]) * 180 / np.pi - 90

                    remajconv = np.sqrt(np.square(re) + np.square(args.psffwhm/2))
                    reminconv = np.sqrt(np.square(re*axrat) + np.square(args.psffwhm / 2))
                    axratconv = reminconv/remajconv

                    objref = galsim.Sersic(
                        n=nser, half_light_radius=remajconv*np.sqrt(axratconv), flux=1,
                        gsparams=galsim.GSParams(realspace_relerr=1e-6, realspace_abserr=1e-10)).shear(
                            g=(1-axratconv)/(1+axratconv), beta=ang * galsim.degrees)
                    imgref = objref.drawImage(imgref, method="real_space")
                else:
                    objref = galsim.Convolve(objref, objpsf)
                    imgref = objref.drawImage(imgref)
            else:
                imgref = objref.drawImage(imgref, method="real_space")
            imgrefarr = imgref.array

        proresult = time_pyprofit(imgref=imgrefarr, width=width*args.oversample, height=height*args.oversample, profiles=profiles, **evalargs)
        if args.galsim:
            img = galsim.ImageD(width, height, scale=1)
            # TODO: Surely there must be a way to change only the gsparams?
            obj = galsim.Sersic(n=nser+(np.random.uniform()-0.5)*1e-8, half_light_radius=re * np.sqrt(axrat), flux=1,
                gsparams=galsim.GSParams(maximum_fft_size=12 * 4096, realspace_relerr=1e-3,
                realspace_abserr=1e-5, integration_relerr=1e-4, integration_abserr=1e-6,
                kvalue_accuracy=1e-4, xvalue_accuracy=1e-4)).shear(
                    g=(1 - axrat) / (1 + axrat), beta=ang * galsim.degrees)

            if label == "CPU":
                for method in methods_galsim:
                    if args.psffwhm > 0:
                        benchobj = galsim.Convolve(obj, objpsf, real_space=(method == "real_space"))
                    else:
                        benchobj = obj
                    gsresult = time_galsim(benchobj, imgref=imgrefarr, image=img, method=method)
                    times[labels_galsim[method]].append(gsresult)

        times[label].append(proresult)

        print("Done nser={:.3e} ang={:.3e} axrat={:.3e} re={:.2e} box={:.2e}".format(nser, ang, axrat, re, box))

    print(" done! (%.3f [s])" % (time.time() - start))

# Print values and exit
errors = []


def value(x):
    # Normal values (from the parameter space)
    if not hasattr(x, 'exception'):
        return "{:6.2f}".format(x)
    # Measurements (which might be errors)
    if x.exception:
        ret = "[E%d]" % len(errors)
        errors.append(x.exception)
        return "%8s" % ret
    return "{:.7e} {:.7e} {:.7e}".format(x.time, x.errorabs, x.negsum)


labels_print = labels + list(labels_galsim_short.values())
labels += list(labels_galsim.values())
out = " ".join(['nser  ', '   ang', ' axrat', '   re', '   box'] + ["{:13s}".format(l + p) for l in labels_print for p in ["_t", "_e", "_n"]])
print(out)
if fileout is not None:
    fileout.write(out + "\n")
for vals in zip(itertools.product(*parameters), *[times[x] for x in labels]):
    vals = list(vals[0]) + list(vals[1:])
    out = " ".join(value(x) for x in vals)
    print(out)
    if fileout is not None:
        fileout.write(out + "\n")

if fileout is not None:
    fileout.close()

if errors:
    print("\nErrors found:")
    for i, e in enumerate(errors):
        print("  E%d: %s" % (i, str(e)))