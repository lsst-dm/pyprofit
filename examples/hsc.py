# Boilerplate up here

import pyprofit.python.profit as pro

import argparse
import copy
import fitsio
import galsim
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import select
from skimage import measure
import subprocess
from subprocess import PIPE
import time


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


# Params: a dict by profile type
def fitimage(image, invmask, inverr, psf, params, plotinit=None,
             printfinal=None, engine=None, constraints=None, **kwargs):
    if plotinit is None:
        plotinit = False
    if printfinal is None:
        printfinal = True

    # A dict split by param type (init, sigma, etc.) and then by profile type
    paramssplit = {}
    for profilename, profileinfo in params.items():
        nprofiles = len(profileinfo["init"])

        # Store the number of params per profile type
        # In retrospect, this could/should be the number of profiles instead. Oh well.
        profiles = [{} for _ in range(nprofiles)]
        for key, value in profileinfo.items():
            # If this is true, each profile of this type has its own values; otherwise copy [0]
            valueperprofile = len(value) > 1
            for i, profile in enumerate(profiles):
                profile[key] = copy.copy(value[i*valueperprofile])

        # Convert the profile list to a parameter array
        profileparams = pro.profiles_to_params(profiles, profilename)
        for key, value in profileparams.items():
            if key not in paramssplit:
                paramssplit[key] = {}
            paramssplit[key][profilename] = value

    # Set up the initial structure that will hold all the data needed afterwards
    data = pro.setup_data(
        0, image, invmask, inverr, psf, **paramssplit, engine=engine, constraints=constraints)

    _, modelim0 = pro.make_image(data.init, data, use_mask=False)

    if plotinit:
        pro.plot_image_comparison(data.image, modelim0, data.invsigim, data.region)
        plt.show()

    # Go, go, go!
    data.verbose = False
    tinit = time.time()
    paramsbest = pro.fit(data, **kwargs)
    timerun = time.time() - tinit

    if printfinal:
        print("Elapsed time: {:.1f}".format(timerun))
        print("Final likelihood: {:.2f}".format(pro.like_model(paramsbest, data)))
        print("Parameter names: " + ",".join(["{:10s}".format(i) for i in data.names[data.tofit]]))
        print("Scaled parameters: " + ",".join(["{:.4e}".format(i) for i in paramsbest]))
        paramsraw = paramsbest * data.sigmas[data.tofit]
        print("Parameters (logged): " + ",".join(["{:.4e}".format(i) for i in paramsraw]))
        paramsraw[data.tolog[data.tofit]] = 10 ** paramsraw[data.tolog[data.tofit]]
        print("Parameters (unlogged): " + ",".join(["{:.4e}".format(i) for i in paramsraw]))

    return paramsbest, timerun, data


def testhsc():
    parser = argparse.ArgumentParser(description="PyProFit fitting GAMA galaxy")

    algo_default = "lbfgs (pygmo), L-BFGS-B (scipy)"
    algos_lib_default = { "scipy": "L-BFGS-B", "pygmo" : "lbfgs" }
    optlibs = ["pygmo", "scipy"]
    flags = {
        "gid":       {"type": str,  "default": "G265911", "desc": "GAMA ID"}
        , "optlib":  {"type": str,  "default": "scipy", "desc": "Optimization library", "values" : optlibs}
        , "algo":    {"type": str,  "default": algo_default, "desc": "Optimization algorithm"}
        , "grad":    {"type": bool, "default": False, "desc": "Use numerical gradient (pygmo)"}
        , "radec":   {"type": str,  "default": None, "desc": "RA,dec string (deg.)", "nargs": 2}
        , "band":    {"type": str,  "default": "HSC-R", "desc": 'HSC Band (g, r, i, z, y)'}
        , "size":    {"type": str,  "default": "20", "desc": 'Cutout half-size (asecs)'}
        , "fitpsf":  {"type": bool, "default": False, "desc": "Fit the PSF first"}
        , "galsim":  {"type": bool, "default": False, "desc": "Use galsim for modeling"}
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
    if args.algo == algo_default:
        args.algo = algos_lib_default[args.optlib]

    engine = None
    if args.galsim:
        engine = "galsim"

    image, psf = gethsc(
        [args.band], args.radec[0], args.radec[1], semiwidth=args.size, semiheight=args.size,
        prefix='-'.join(args.radec)
    )
    try:
        fitpsf = False
        if fitpsf:
            cenx, ceny = [x / 2.0 for x in psf.shape]
            mag = -2.5 * np.log10(1/2)
            size = np.sqrt(cenx ** 2.0 + ceny ** 2.0) / 4

            params = {
                "moffat": {
                    "init": [np.array([cenx,   ceny,   mag, size,   1/2, 0, 0.9, 0]),
                             np.array([np.nan, np.nan, mag, size/2, 1/5, 90, 0.8, 0])]
                    , "tofit": [np.array([True, True, True, True,   False, True, True, False]),
                                np.array([False, False, False, True, False, True, True, False])]
                    , "tolog": [np.array([False, False, False, True, True, False, True, False])]
                    , "sigmas": [np.array([2, 2, 5, 1, 1, 30, 0.3, 0.3])]
                    , "lowers": [np.array([0.,                      0,    0,  0.1, 0.1, -180, -1,   -1])]
                    , "uppers": [np.array([psf.shape[0], psf.shape[1], np.inf, 10,  0.9,  360, -1e-3, 1])]
                }
            }

            # Yes this is ugly. Deal w/it
            def constraints(profiles):
                profiles["moffat"][1]["mag"] = -2.5*np.log10(1-10**(-0.4*profiles["moffat"][0]["mag"]))
                for profile in profiles["moffat"]:
                    profile["con"] = 1/profile["con"]
                return profiles

            imagegs = galsim.ImageD(psf, scale=1)
            shapelet = galsim.Shapelet.fit(5, 10, imagegs)

            paramsbest, timerun, data = fitimage(
                psf, psf*0+1, psf*0+1e3, None, params, plotinit=True, engine=engine, constraints=constraints,
                **{param: getattr(args, param) for param in ["algo", "grad", "optlib"]},
            )

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
                , "uppers": [np.array([nxy[0], nxy[1], 30, 2,  np.log10(4),  360, -1e-3, 1])]
            }
        }

        paramsbest, timerun, data = fitimage(
            image["data"], image["invmask"], image["inverr"], psf, params, engine=engine,
            **{param: getattr(args, param) for param in ["algo", "grad", "optlib"]}
        )

        all_params, modelim  = pro.make_image(paramsbest, data, use_mask=False)
        pro.plot_image_comparison(data.image, modelim, data.invsigim, data.region)
        plt.show()

#        input("Press Enter to continue...")

    except Exception as error:
        print(type(error))
        print(error.args)
        print(error)


if __name__ == '__main__':
    hsc.testhsc()
