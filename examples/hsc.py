# Boilerplate up here

import pyprofit.python.profit as pro

import argparse
import fitsio
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import select
from skimage import measure
import subprocess
from subprocess import PIPE

algo_default = "lbfgs (pygmo), L-BFGS-B (scipy)"
algos_lib_default = {"scipy": "L-BFGS-B", "pygmo": "lbfgs"}
optlibs = ["pygmo", "scipy"]

# Shamelessly stolen from astrometry.net
# Returns (rtn, out, err)
def run_command(cmd, timeout=None, callback=None, stdindata=None,
                tostring=True):
    """
    Run a command and return the text written to stdout and stderr, plus
    the return value.

    In python3, if *tostring* is True, the output and error streams
    will be converted to unicode, otherwise will be returned as bytes.

    Returns: (int return value, string out, string err)
    """
    child = subprocess.Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
    (fin, fout, ferr) = (child.stdin, child.stdout, child.stderr)

    stdin = fin.fileno()
    stdout = fout.fileno()
    stderr = ferr.fileno()
    outdata = []
    errdata = []
    ineof = outeof = erreof = False
    block = 1024
    while True:
        readers = []
        writers = []
        if not ineof: writers.append(stdin)
        if not outeof: readers.append(stdout)
        if not erreof: readers.append(stderr)
        if not len(readers):
            break
        (ready_readers, ready_writers, _) = select.select(readers, writers, [], timeout)
        if stdin in ready_writers and stdindata:
            bytes_written = os.write(stdin, stdindata[:block])
            stdindata = stdindata[bytes_written:]
            if not stdindata:
                fin.close()
                ineof = True
        if stdout in ready_readers:
            outchunk = os.read(stdout, block)
            if len(outchunk) == 0:
                outeof = True
            outdata.append(outchunk)
        if stderr in ready_readers:
            errchunk = os.read(stderr, block)
            if len(errchunk) == 0:
                erreof = True
            errdata.append(errchunk)
        if callback:
            callback()
    fout.close()
    ferr.close()
    w = child.wait()
    if py3:
        out = b''.join(outdata)
        err = b''.join(errdata)
        if tostring:
            out = out.decode()
            err = err.decode()
    else:
        out = ''.join(outdata)
        err = ''.join(errdata)

    if not os.WIFEXITED(w):
        return (-100, out, err)
    rtn = os.WEXITSTATUS(w)
    return (rtn, out, err)


def readpasshsc(file):
    parser = configparser.ConfigParser(delimiters=[" "])

    with open(file) as text:
        text = itertools.chain(("[top]",), text)
        parser.read_file(text)

    return parser["top"]["User"], parser["top"]["Password"]


# For how to set your login from a file, see:
#   https://serverfault.com/questions/267470/wget-is-there-a-way-to-pass-username-and-password-from-a-file


def downloadhsc(outfile, ra, dec, semiwidth=None, semiheight=None, getpsf=None, imgtype=None, hscfilter=None,
                getimage=None, getmask=None, getvariance=None, rerun=None, curl=None, user=None, password=None):
    if imgtype is None:
        imgtype = "coadd"
    if hscfilter is None:
        hscfilter = "HSC-R"
    if rerun is None:
        rerun = "pdr1_wide"
    if getimage is None:
        getimage = True
    if getmask is None:
        getmask = True
    if getvariance is None:
        getvariance = True
    if curl is None:
        curl = False
    if getpsf is None:
        getpsf = False

    values = {
        "ra": ra
        , "dec": dec
        , "type": imgtype
        , "filter": hscfilter
        , "rerun": rerun
    }

    if not getpsf:
        if semiwidth is None:
            semiwidth = "20asec"

        if semiheight is None:
            semiheight = "20asec"

        values["sw"] = semiwidth
        values["sh"] = semiheight

        if getimage:
            values["image"] = "on"

        if getmask:
            values["mask"] = "on"

        if getvariance:
            values["variance"] = "on"

    flags = ['{:s}={:s}'.format(key, value) for key, value in values.items()]
    flags = "&".join(flags)

    url = "https://hsc-release.mtk.nao.ac.jp/"

    if getpsf:
        url += "psf/pdr1/cgi/getpsf?"
    else:
        url += "das_quarry/cgi-bin/quarryImage?"

    url += flags

    cmd = ""
    if curl:
        cmd += "curl -n -o "
    else:
        cmd += "wget --continue -nv -O "

    fmt = "'{:s}' '{:s}'"
    args = [outfile, url]
    if user is not None:
        fmt += " --user '{:s}"
        if not curl:
            fmt += "'"
        args.append(user)
    if password is not None:
        if curl:
            fmt += ":{:s}'"
        else:
            fmt += " --password '{:s}'"
        args.append(password)

    cmd += fmt.format(*args)

    (rtn, out, err) = run_command(cmd)
    if rtn:
        print('Command failed: command', cmd)
        print('Output:', out)
        print('Error:', err)
        print('Return val:', rtn)
        return None

    return rtn


def gethsc(bands, ra, dec, prefix=None, **kwargs):

    bandshsc = ["HSC-" + band for band in ["G", "R", "I", "Z", "Y"]] + ["NB" + ang for ang in ["0816", "0921"]]
    extensions = {
        "image": "IMAGE"
        , "mask": "MASK"
        , "variance": "VARIANCE"
    }

    for idxband, bandname in enumerate(bands):
        if bandname not in bandshsc:
            raise ValueError("Band " + bandname + " not in HSC bands " + ','.join(bandshsc))
        prefixband = list(filter(None, ["HSC-wide", prefix, bandname]))
        fitsband = "_".join(prefixband) + ".fits"
        if not os.path.isfile(fitsband):
            rtn = downloadhsc(fitsband, ra, dec, hscfilter=bandname, **kwargs)

        headerimage = fitsio.read_header(fitsband)
        fitsobj = fitsio.FITS(fitsband)
        for extension, extensionname in extensions.items():
            if extensionname not in fitsobj:
                raise RuntimeError(
                    "Couldn't find {:s} extension='{:s}' in file='{:s}'",
                    extension, extensionname, fitsband)

        psfband = "_".join(prefixband + ["psf"]) + ".fits"
        if not os.path.isfile(psfband):
            rtn = downloadhsc(psfband, ra, dec, getpsf=True, hscfilter=bandname)

        fluxscale = headerimage["FLUXMAG0"]
        data = fitsio.read(fitsband, ext=extensions["image"])/fluxscale

        # Read the mask - it's not a segmentation map, but we can at least select all contiguous "detected"
        # pixels assuming the object is in the center
        invmask = fitsio.read(fitsband, ext=extensions["mask"])
        headermask = fitsio.read_header(fitsband, ext=extensions["mask"])
        cen = [np.int(np.floor(x/2.0)) for x in invmask.shape]
        # Bright objects that are not saturated and should be safe to use
        brightflag = 2**headermask["MP_BRIGHT_OBJECT"]
        # detected = 2**headermask["MP_DETECTED"]
        # This will also select blended sources, but it's the best you can do with no segmentation map
        segments = measure.label(invmask, background=0)
        invmask = 1*(segments == segments[cen[0], cen[1]]) | 1*(invmask == 0) | 1*(invmask == brightflag) != 0
        # The scaling below will get it into maggies
        variance = fitsio.read(fitsband, ext=extensions["variance"])
        psf = fitsio.read(psfband)

        image = {
            "data": data
            , "invvar": fluxscale**2/variance
            , "inverr": fluxscale/np.sqrt(variance)
            , "invmask" : invmask
            , "header": fitsio.read_header(fitsband, extensions["image"])
            , "name": '_'.join(["coadd", bandname])
        }
        return image, psf


def getlimits(param, profile, log):
    limits = None
    if profile == "moffat":
        if param == "con":
            limits = [1.11, 10]
    elif profile == "sersic":
        if param == "nser":
            # TODO: get rid of this and other ugliness by allowing users to set transformed limits
            limits = [0.3+log*4e-17, 6.0]
    if log:
        limits = np.log10(limits)
    return limits


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def testhsc(radec=None, band=None, size=None, psffit=False, psfmodel=None, psfmodeluse=False, optlib="scipy",
            algo=None, grad=False, galsim=False, useobj=False):
    if algo is None:
        algo = algos_lib_default[optlib]

    engine = None
    if galsim:
        engine = "galsim"

    image, psf = gethsc(
        [band], radec[0], radec[1], semiwidth=size, semiheight=size,
        prefix='-'.join(radec)
    )
    try:
        if psffit:
            psfmodel = psfmodel.split(",")[0].split(":")
            psfprofile = psfmodel[0]
            psfncomps = np.int(psfmodel[1])
            cenx, ceny = [x / 2.0 for x in psf.shape]

            isgaussian = psfprofile == "gaussian"

            size = np.sqrt(cenx ** 2.0 + ceny ** 2.0) / (4 + 0*isgaussian)
            # This is going to be the fraction of the remaining flux in the previous component
            # (Confusing, I know, but it lets you fit the magnitude in the first component's mag)
            fluxfrac = 1/np.arange(psfncomps, 0, -1)

            if useobj:
                pass

            if isgaussian:
                psfprofile = "sersic"

            if psfprofile == "moffat":
                shapename = "con"
                shapelog = False
                shapelimits = np.flip(1./np.array(getlimits(shapename, psfprofile, shapelog)), 0)
                shapes = [1./5.] + list(np.repeat(1./2.5, psfncomps-1.))
            elif psfprofile == "sersic":
                shapename = "nser"
                shapelog = True
                shapelimits = getlimits(shapename, psfprofile, shapelog)
                shapes = np.repeat(0.5, psfncomps)
            else:
                shapename = ""
                shapelimits = getlimits("", psfprofile)
                shapes = None
                shapelog = False

            init = [np.array([cenx, ceny, 0, size/2, shapes[0], 0, 0.9, 0])]
            tofit = [np.array([True, True, False, True, False, True, True, False])]
            for comp in range(1, psfncomps, 1):
                # These are obviously not very clever initial guesses
                # One idea might be to add components one-by-one based on the residuals of the
                # single-component fit
                init += [np.array([np.nan, np.nan, np.log10(fluxfrac[comp-1]), size,
                         shapes[comp], 90*comp/psfncomps, 0.8, 0])]
                tofit += [np.array([False, False, True, True, False, True, True, False])]

            params = {
                psfprofile: {
                    "init": init
                    , "tofit": tofit
                    , "tolog": [np.array([False, False, False, True, shapelog, False, True, False])]
                    , "sigmas": [np.array([2, 2, 5, 1, 1, 30, 0.3, 0.3])]
                    , "lowers": [np.array([0.,           0,            -np.inf, 1e-3,
                                           shapelimits[0], -180, -1,   -1])]
                    , "uppers": [np.array([psf.shape[0], psf.shape[1], -0.1,       1e2,
                                           shapelimits[1],  360, -1e-4, 1])]
                }
            }

            # Yes this is ugly. It's the best way at the moment to fit a constrained total magnitude
            # It should work for an unconstrained one too, but is that better than fitting components mags?
            def constraints(cprofiles):
                profilename = list(cprofiles.keys())[0]
                comps = cprofiles[profilename]
                ncomps = len(comps)
                logflux = -0.4*comps[0]["mag"]

                for compidx in range(1, ncomps, 1):
                    comps[compidx-1]["mag"] = -2.5*(logflux + comps[compidx]["mag"])
                    logflux = np.log10(1-10**(comps[compidx]["mag"]+logflux))

                comps[ncomps-1]["mag"] = -2.5*logflux

                if profilename == "moffat":
                    for profile in cprofiles[profilename]:
                        profile[shapename] = 1.0/profile[shapename]

                return cprofiles

#            imagegs = gs.ImageD(psf, scale=1)
#            shapelet = gs.Shapelet.fit(5, 10, imagegs)

            paramsbest, paramstransformed, paramslinear, timerun, data = pro.fit_image(
                psf, psf*0+1, psf*0+np.prod(psf.shape), None, params, plotinit=True,
                constraints=constraints, use_allpriors=True, method="fft",
                optlib=optlib, algo=algo, grad=grad
            )

            if not isgaussian:
                offset = 0
                profiles_rebuilt, allparams, _ = pro.data_rebuild_profiles(
                    paramsbest, data, applyconstraints=False)
                for idx, tofitcomp in enumerate(params[psfprofile]["tofit"]):
                    initcomp = params[psfprofile]["init"][idx]
                    sumtofit = np.sum(tofitcomp)
                    # TODO: This is hideous - is there a better way before adding Source/Model classes?
                    # Selecting only tofitcomp isn't really necessary. Could check if not tofitcomp
                    # are identical
                    initbest = np.array(list(profiles_rebuilt[psfprofile][idx].values())[0:len(initcomp)])
                    initcomp[tofitcomp] = initbest[tofitcomp]
                    # Nans are inherited so skip checking them
                    compstocheck = ~tofitcomp & ~np.isnan(initcomp)
                    if not np.all(initcomp[compstocheck] == initbest[compstocheck]):
                        raise RuntimeError("initcomp[compstocheck]=" + str(initcomp[compstocheck]) + \
                            " differs from initbest[compstocheck]=" + str(initbest[compstocheck]) + \
                            " but these should not have changed after fitting"
                        )
                    offset += sumtofit
                    tofitcomp[4] = True
                paramsbest, paramstransformed, paramslinear, timerun, data = pro.fit_image(
                    psf, psf*0+1, psf*0+np.prod(psf.shape), None, params, plotinit=False,
                    constraints=constraints, use_allpriors=True, method="fft",
                    optlib=optlib, algo=algo, grad=grad
                )

            verifyreal = False
            if verifyreal:
                data.method = "real_space"
                paramsbest, paramstransformed, paramslinear, timerun, data = pro.fit_data(
                    data, init=paramsbest, optlib=optlib, algo=algo, grad=grad,
                )

                if psfmodeluse:
                    if engine == "galsim":
                        profiles, _, _ = pro.data_rebuild_profiles(paramsbest, data)
                        psf = pro.make_model_galsim(profiles, None, None, None)
                    else:
                        psfparams, psf = pro.make_image(paramsbest, data, False)

        nxy = image["data"].shape
        cenx, ceny = [x/2.0 for x in nxy]
        mag = -2.5*np.log10(np.sum(image["data"][image["invmask"]])/2)
        size = np.sqrt(cenx**2.0 + ceny**2.0)/4

        # Initial parameter guess - ellipses can be done better
        params = {
            "sersic": {
                "init": [np.array([cenx, ceny, mag, size, 0.60, 130, 0.5, 0]),
                         np.array([np.nan, np.nan, mag, size / 3, 1.15, 120, 0.7, 0])]
                , "tofit": [np.array([True, True, True, True, True, True, True, False]),
                            np.array([False, False, True, True, True, True, True, False])]
                , "tolog": [np.array([False, False, False, True, True, False, True, False])]
                , "sigmas": [np.array([2,           2,  5, 1,  1,   30, 0.3, 0.3])]
                , "lowers": [np.array([0.,          0, 10, 0,  np.log10(0.5), -180,  -1, -1])]
                , "uppers": [np.array([nxy[0], nxy[1], 30, 2,  np.log10(4),  360, -1e-4, 1])]
            }
        }

        paramsbest, paramstransformed, paramslinear, timerun, data = pro.fit_image(
            image["data"], image["invmask"], image["inverr"], psf, params, engine=engine,
            use_allpriors=True, method="fft", plotinit=True,
            optlib=optlib, algo=algo, grad=grad
        )

        all_params, modelim  = pro.make_image(paramsbest, data, use_calcinvmask=False)
        pro.plot_image_comparison(data.image, modelim, data.invsigim, data.region)
        plt.show()

#        input("Press Enter to continue...")

    except Exception as error:
        print(type(error))
        print(error.args)
        print(error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyProFit fitting GAMA galaxy")

    flags = {
        "radec":     {"type": str, "default": None, "desc": "RA,dec string (deg.)", "nargs": 2}
        , "band": {"type": str, "default": "HSC-R", "desc": 'HSC Band (g, r, i, z, y)'}
        , "size": {"type": str, "default": "20", "desc": 'Cutout half-size (asecs)'}
        , "psffit": {"type": str2bool, "default": False, "nargs": '?', "desc": "Fit the PSF first"}
        , "psfmodeluse":
            {"type": str2bool, "default": False, "desc": "Use the fitted PSF model for galaxy fitting"}
        , "psfmodel":
            {"type": str, "default": "gaussian:1", "desc":
                "PSF model description as comma-separated list of [profile]:[number];"
                "only one profile type currently supported"}
        , "optlib":    {"type": str,  "default": "scipy", "desc": "Optimization library", "values": optlibs}
        , "algo":      {"type": str,  "default": None, "desc": "Optimization algorithm"}
        , "grad":      {"type": str2bool, "default": False, "desc": "Use numerical gradient (pygmo)"}
        , "galsim":    {"type": str2bool, "default": False, "desc": "Use galsim for modeling"}
    }

    for key, value in flags.items():
        helpstr = value["desc"] + "; default=(" + str(value["default"]) + ")"
        if "values" in value:
            helpstr += "; allowed values=(" + ",".join(value["values"]) + ")"
        if "nargs" in value:
            nargs = value["nargs"]
        else:
            nargs = "?"
        parser.add_argument("-" + key, type=value["type"], nargs=nargs, help=helpstr, default=value["default"])

    args = parser.parse_args()
    testhsc(**vars(args))
