import argparse
import inspect
import numpy as np

import pyprofit.python.objects as proobj
import pyprofit.python.util as proutil

options = {
    "algo":        {"default": {"scipy": "L-BFGS-B", "pygmo": "lbfgs"}},
    "background":  {"default": [1.e3]},
    "engine":      {"avail": ["galsim", "libprofit"], "default": "galsim"},
    "galaxyflux":  {"default": [1.e5]},
    "galaxyradius":     {"default": [5.]},
    "galaxymodel":      {"default": "sersic:1"},
    "galaxycenoffsets": {"default": [0., 0.]},
    "imagesize":   {"default": [60, 60]},
    "optlib":      {"avail": ["pygmo", "scipy"], "default": "scipy"},
    "psfmodel":    {"default": "gaussian:1"},
    "psfflux":     {"default": [1.e4]},
    "psfradius":   {"default": [4.]},
    "psfsize":     {"default": [21, 21]},
    "psffit":      {"default": False},
    "psfmodeluse": {"default": False},
}


def testfitting(
    bands=["1"], imagesize=options["imagesize"]["default"],
    background=options["background"]["default"],
    galaxyflux=options["galaxyflux"]["default"],
    galaxyradius=options["galaxyradius"]["default"],
    galaxymodel=options["galaxymodel"]["default"],
    galaxycenoffsets=options["galaxycenoffsets"]["default"],
    psfsize=options["psfsize"]["default"],
    psfmodel=options["psfmodel"]["default"],
    psfflux=options["psfflux"]["default"],
    psfradius=options["psfradius"]["default"],
    psffit=options["psffit"]["default"],
    psfmodeluse=options["psfmodeluse"]["default"],
    optlib=options["optlib"]["default"],
    algo=options["algo"]["default"][options["optlib"]["default"]],
    engine=options["engine"]["default"]
):
    bandfluxespsf = dict(zip(bands, psfflux))
    modelpsf = proutil.getmodel(bandfluxespsf, psfmodel, psfsize, psfradius, [1.0], [0], engine="galsim")
    modelpsf.evaluate(keepimages=True, getlikelihood=False)

    if psffit:
        pass

    bandfluxesgal = dict(zip(bands, galaxyflux))
    model = proutil.getmodel(bandfluxesgal, galaxymodel, imagesize, galaxyradius, [0.5], [0], engine="galsim")
    model.evaluate(keepimages=True, getlikelihood=False)
    for band, exposures in model.data.exposures.items():
        exposures[0].meta["modelimagetrue"] = np.array(exposures[0].meta["modelimage"])
        if psfmodeluse:
            # TODO: Check that normalize flux works
            for flux in modelpsf.sources[0].photometricmodel.fluxes:
                flux.value = 0
            psfband = proobj.PSF(band=band, model=modelpsf.sources[0])
        else:
            imagepsf = proutil.absconservetotal(np.copy(modelpsf.data.exposures[band][0].meta["modelimage"]))
            imagepsf /= np.sum(imagepsf)
            psfband = proobj.PSF(band=band, image=proutil.absconservetotal(imagepsf))
        for exposure in exposures:
            exposure.psf = psfband
            exposure.meta["sky"] = np.random.poisson(background, exposure.image.shape)

    backgroundsband = dict(zip(bands, background))
    # Now get the convolved model
    model.evaluate(keepimages=True, getlikelihood=False)
    for band, exposures in model.data.exposures.items():
        for exposure in exposures:
            exposure.meta["modelimage"] = proutil.absconservetotal(exposure.meta["modelimage"])
            exposure.image = np.float64(np.random.poisson(exposure.meta["modelimage"] + exposure.meta["sky"]))
            exposure.sigmainverse = np.power(exposure.image, -0.5)
            # Assume optimal sky subtraction
            exposure.image -= backgroundsband[band]

    # Plot the idealized model
    model.evaluate(plot=True)

    if engine != "galsim":
        model.engine = engine
        model.evaluate(plot=True)
    modellibopts = {
        "algo": algo,
    }
    modeller = proobj.Modeller(model=model, modellib=optlib, modellibopts=modellibopts)
    modeller.fit(timing=True, printfinal=True, printsteps=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyProFit single model PSF/galaxy fitting test")

    signature = inspect.signature(testfitting)
    defaults = {
        k: v.default for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    flags = {
        "background": {"type": float, "default": defaults["background"],
                       "desc": 'Background counts (pixels^-1)'},
        "psfsize": {"type": int, "nargs": 2, "default": defaults["psfsize"],
                    "desc": 'PSF image size (pixels)'},
        "psfflux": {"type": float, "desc": 'PSF flux (counts)'},
        "psfradius": {"type": float, "desc": 'PSF FWHM (pixels)'},
        "psfmodel": {"type": str, "desc": "PSF model description as comma-separated "
                     "list of [profile]:[number]"},
        "imagesize": {"type": int, "nargs": 2, "desc": 'Galaxy image size (pixels)'},
        "galaxyflux": {"type": float, "desc": 'Galaxy flux (counts)'},
        "galaxycenoffsets": {"type": float, "nargs": 2, "desc": 'Galaxy center offset (pixels)'},
        "galaxymodel": {"type": str, "desc": "Galaxy model description as comma-separated "
                  "list of [profile]:[number]"},
        "psffit": {"type": proutil.str2bool, "desc": "Fit the PSF first"},
        "psfmodeluse": {"type": proutil.str2bool, "desc": "Use the fitted PSF model for galaxy fitting"},
        "optlib":    {"type": str,  "default": "scipy", "desc": "Optimization library", "values": options[
            "engine"]["avail"]},
        "algo":      {"type": str,  "default": None, "desc": "Optimization algorithm"},
        "engine":    {"type": str, "default": None, "desc": "Use galsim for modeling"},
    }

    for key, value in flags.items():
        if key in defaults:
            default = defaults[key]
        else:
            default = value["default"]
        helpstr = value["desc"] + "; default=(" + str(default) + ")"
        if "values" in value:
            helpstr += "; allowed values=(" + ",".join(value["values"]) + ")"
        if "nargs" in value:
            nargs = value["nargs"]
        else:
            nargs = "?"
        parser.add_argument("-" + key, type=value["type"], nargs=nargs, help=helpstr, default=default)

    args = parser.parse_args()
    testfitting(**vars(args))
