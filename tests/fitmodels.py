import argparse
import copy
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import pyprofit.python.objects as proobj
import pyprofit.python.util as proutil

options = {
    "algos":       {"default": {"scipy": "L-BFGS-B", "pygmo": "lbfgs"}},
    "backgrounds": {"default": [1.e3]},
    "engines":     {"avail": ["galsim", "libprofit"], "default": "galsim"},
    "bands":       {"default": [""]},
    "galaxyfluxes":     {"default": [1.e5]},
    "galaxyfluxmults":  {"default": [1.]},
    "galaxyradii":      {"default": [5.]},
    "galaxycenoffsets": {"default": [[0., 0.], [-0.5, -0.5]]},
    "imagesizes":  {"default": [60]},
    "optlibs":     {"avail": ["pygmo", "scipy"], "default": ["scipy"]},
    "psfaxrats":   {"default": [0.95]},
    "psffluxes":   {"default": [1.e4]},
    "psfradii":    {"default": [4.]},
    "psfsizes":    {"default": [21]},
    "psffit":      {"default": False},
    "psfmodeluse": {"default": False},
    "psfoptlib":   {"default": "pygmo"},
    "psfoptalgo":  {"default": "neldermead"},
}


def getparamvalues(model, transformed=True):
    return [x.getvalue(transformed=transformed) for x in model.getparameters()]


# TODO: Make this a member function of model?
def modelpoissonsample(exposuresbyband, backgroundsbyband):
    for band, exposures in exposuresbyband:
        background = backgroundsbyband[band]
        for exposure in exposures:
            image = proutil.absconservetotal(np.array(exposure.meta['modelimage']))
            image = np.float64(np.random.poisson(image) + np.random.poisson(background, image.shape))
            exposure.sigmainverse = np.power(image, -0.5)
            exposure.image = image - background


# TODO: Make this a member function of model?
def modelnormalize(model):
    fluxparams = [param for param in model.getparameters() if isinstance(param, proobj.FluxParameter) and
                  not param.isfluxratio]
    fluxtotal = {}
    for param in fluxparams:
        band = param.band
        if band not in fluxtotal:
            fluxtotal[band] = 0.
        fluxtotal[band] += param.getvalue(transformed=False)
    for param in fluxparams:
        param.setvalue(param.getvalue(transformed=False)/fluxtotal[param.band])


def setpsf(model, psf, usemodel=True):
    for band, exposures in model.data.exposures.items():
        if usemodel:
            for exposure in exposures:
                exposure.psf = proobj.PSF(band=band, model=psf.sources[0])
        else:
            # TODO: Check if same length
            for exposure, exposurespsf in zip(exposures, psf.data.exposures[band]):
                image = proutil.absconservetotal(np.copy(exposurespsf.image))
                exposure.psf = proobj.PSF(band=band, image=image/np.sum(image))


# I apologize for the naming of the PSF params. psf is the actual PSF, psfmodel is the model for testing
# fitting the wrong PSF model. usemodel for both specifies whether the PSF is a Model or not
def testfitting(models, paramsbymodeltype, psf=None, psfusemodel=False, psfmodel=None, psfmodelusemodel=True,
                backgroundsbyband={"": options["backgrounds"]["default"]},
                engine=options["engines"]["default"], optlib=options["optlibs"]["default"][0],
                optlibopts=None, plot=False, printsteps=None, timing=False,
                psfname=None, psfmodelname=None):
    engineoptsgsreal = {"drawmethod": "real_space"}
    results = []
    if psf is not None:
        if psfname is None:
            psfname = ""
        if psfmodel is not None:
            psfmodelname = ""

    # The true model
    for modeltype, paramsoftype in paramsbymodeltype.items():
        model = models[modeltype]

        # The model params
        for params in paramsoftype:
            # TODO: Add a check to the zip
            # Set params to the right values
            for dest, valuenew in zip(model.getparameters(), params):
                dest.setvalue(valuenew, transformed=True)
            if psf is not None:
                setpsf(model, psf, usemodel=psfusemodel)
            model.evaluate(keepimages=True, getlikelihood=False, engineopts=engineoptsgsreal)
            modelpoissonsample(model.data.exposures.items(), backgroundsbyband)
            if psfmodel is not None:
                setpsf(model, psfmodel, usemodel=psfmodelusemodel)

            # The model to fit
            for modeltypefit, modelfit in models.items():
                print("Fitting object={:s} with model={:s}".format(modeltype, modeltypefit))
                modelfit.data = model.data
                modelfit.engine = engine
                modeller = proobj.Modeller(model=modelfit, modellib=optlib, modellibopts=optlibopts)
                # GalSim seems to have issues with Moffat profiles even with limits to the concentration
                # Real space integration seems safer
                if engine == "galsim":
                    hasmoffat = "moffat" in modeltypefit
                    if hasmoffat:
                        #engineold = modelpsf.engine
                        #modelpsf.engine = "libprofit"
                        engineold = modelfit.engineopts
                        modelfit.engineopts = engineoptsgsreal
                result = modeller.fit(printsteps=printsteps, printfinal=True, timing=timing)
                if plot:
                    modeller.evaluate(plot=True)
                    title = "Source={} Model={}".format(modeltype, modeltypefit)
                    if psfname is not None:
                        title += " PSF={}".format(psfname)
                    if psfmodelname is not None:
                        title += " PSFModel={}".format(psfmodelname)
                    plt.suptitle(title)
                    plt.show()
                    input("Press Enter to continue...")
                if engine == "galsim":
                    if "moffat" in modeltypefit:
                        #modelpsf.engine = engineold
                        modelfit.engineopts = engineold
                result = {"fitresult": result, "fitinfo": modeller.fitinfo}
                results.append(result)
    return results


def gettestfittingmodels(
    bands="", backgrounds=options["backgrounds"]["default"],
    psfsizes=options["psfsizes"]["default"], psffluxes=options["psffluxes"]["default"],
    psfradii=options["psfradii"]["default"], psfaxrats=options["psfaxrats"]["default"],
    imagesizes=options["imagesizes"]["default"], galaxyfluxes=options["galaxyfluxes"]["default"],
    galaxyfluxmults=options["galaxyfluxmults"]["default"], galaxyradii=options["galaxyradii"]["default"],
):
    inputs = [backgrounds, psfsizes, psffluxes, psfradii, psfaxrats, imagesizes, galaxyfluxes]
    inputsreal = [all(np.isreal(x)) for x in inputs]
    if not all(inputsreal):
        # TODO: More useful error
        raise ValueError("Not all gettestfittingmodels inputs real numbers: " + str(inputsreal))
    bandscount = len(bands)
    isnotequalcounts = [len(x) != bandscount for x in
                        [backgrounds, psfsizes, psffluxes, psfradii, psfaxrats, imagesizes, galaxyfluxes]]
    if any(isnotequalcounts):
        # TODO finish
        raise ValueError("All arguments to gettestfittingmodels must be same length but they are:" +
                         str(bandscount))

    backgrounds = np.array(backgrounds)
    psfradii = np.array(psfradii)
    psffluxes = np.array(psffluxes)
    psfaxrats = np.array(psfaxrats)

    # Squarify image/PSF sizes (could add rectangular option/requirement later)
    # Note: We don't support differently sized PSFs and/or images yet
    psfsizes = [[i, i] for i in psfsizes][0]
    imagesizes = [[i, i] for i in imagesizes][0]

    fluxesbyband1 = {band: 1. for band in bands}
    fluxesbyband2 = {band: 0.5 for band in bands}
    angles = np.repeat(90.0, bandscount)
    modelspsf = {
        "gaussian:1": proutil.getmodel(fluxesbyband1, "gaussian:1", psfsizes, psfradii, psfaxrats,
                                       np.repeat(90.0, bandscount)),
        "gaussian:2": proutil.getmodel(fluxesbyband2, "gaussian:2", psfsizes,
                                       np.concatenate((psfradii*1.5, psfradii*0.725)),
                                       np.concatenate((psfaxrats, psfaxrats)),
                                       np.concatenate((angles, angles))),
        "moffat:1":   proutil.getmodel(fluxesbyband1, "moffat:1", psfsizes, psfradii, psfaxrats, angles,
                                       slopes=np.repeat(2.5, bandscount))
    }
    modelspsf = {"gaussian:1": modelspsf["gaussian:1"]}
    psfs = {model: [] for model in modelspsf}

    fluxesbyband1 = {i[0]: i[1] for i in zip(bands, psffluxes)}
    fluxesbyband2 = {i[0]: np.repeat(i[1] / 2.0, 2) for i in zip(bands, psffluxes)}
    # TODO: Is this really easier than just changing the value(s) of the relevant parameters?
    # It's definitely not faster...
    for angle in np.linspace(0., 45., 10):
        angles = np.repeat(angle, bandscount)
        model = "gaussian:1"
        psfs[model].append(getparamvalues(proutil.getmodel(
            fluxesbyband1, model, psfsizes, psfradii, psfaxrats, angles, nexposures=0)))
        model = "gaussian:2"
        if model in modelspsf:
            psfs[model].append(getparamvalues(proutil.getmodel(
                fluxesbyband2, model, psfsizes, np.concatenate((psfradii*2., psfradii*0.5)),
                np.concatenate((psfaxrats, psfaxrats)), np.concatenate((angles, angles)), nexposures=0)))
        # single Moffat with nomin*al 2.5 slope
        model = "moffat:1"
        if model in modelspsf:
            psfs[model].append(getparamvalues(proutil.getmodel(
                fluxesbyband1, model, psfsizes, psfradii, psfaxrats, angles, slopes=np.repeat(2.5, bandscount),
                nexposures=0)))

    galaxyfluxes = np.array(galaxyfluxes)
    galaxyradii = np.array(galaxyradii)
    galaxyfluxmults = np.array(galaxyfluxmults)

    fluxesbyband1 = {band: 1. for band in bands}
    models = {
        "sersic:1": proutil.getmodel(fluxesbyband1, "sersic:1", imagesizes, np.repeat(5., bandscount),
                                     np.repeat(0.5, bandscount), angles, slopes=np.repeat(2., bandscount))
    }
    galaxies = {model: [] for model in models}

    # See Cortese et al. 2016 (or modify to suit your taste)
    diskheightratio = 0.2
    bulgeheightratios = np.array([0.4, 0.6, 0.8, 1.0])
    for fluxmult in galaxyfluxmults:
        fluxesbyband1 = {i[0]: fluxmult*i[1] for i in zip(bands, galaxyfluxes)}
        for radius in galaxyradii:
            radii = np.repeat(radius, bandscount)
            for angle in np.linspace(0., 180., 10):
                angles = np.repeat(angle, bandscount)
                for cosi in np.linspace(0., 1., 10):
                    model = "sersic:1"
                    # See Cortese et al. 2016 eqn. (5)
                    axrat = np.sqrt((cosi * (1-diskheightratio**2))**2 + diskheightratio**2)
                    # exponential disk
                    galaxies[model].append(getparamvalues(proutil.getmodel(
                        fluxesbyband1, "sersic:1", imagesizes, radii, np.repeat(axrat, bandscount),
                        angles, slopes=np.repeat(1., bandscount), nexposures=0)))
                    # Sersic bulge
                    axrats = np.unique(np.sqrt((cosi * (1.-bulgeheightratios**2))**2 + bulgeheightratios**2))
                    for axrat in axrats:
                        for nser in [2., 3., 4.]:
                            galaxies[model].append(getparamvalues(proutil.getmodel(
                                fluxesbyband1, "sersic:1", imagesizes, radii, np.repeat(axrat, bandscount),
                                angles, slopes=np.repeat(nser, bandscount), nexposures=0)))
                    # exp + dev
                    # double Sersic

    rv = {
        "backgroundsbyband": {band: background for band, background in zip(bands, backgrounds)},
        "models": models,
        "modelspsf": modelspsf,
        "galaxies": galaxies,
        "psfs": psfs,
    }
    return rv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyProFit single model PSF/galaxy fitting test')

    signature = inspect.signature(testfitting)
    defaults = {
        k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty
    }

    flags = {
        'bands':      {'type': str,   'nargs': '*', 'help': 'Bandpass filter names'},
        'backgrounds':{'type': float, 'nargs': '*', 'help': 'Background counts (pixels^-1)'},
        'psfsizes':   {'type': int,   'nargs': '*', 'help': 'PSF image sizes (pixels)'},
        'psffluxes':  {'type': float, 'nargs': '*', 'help': 'PSF fluxes [counts]'},
        'psfradii':   {'type': float, 'nargs': '*', 'help': 'PSF D_50 (=FWHM for Gaussians only) [pixels]'},
        'psfaxrats':  {'type': float, 'nargs': '*', 'help': 'PSF ellipticities [fraction]'},
        'imagesizes': {'type': int,   'nargs': '*', 'help': 'Galaxy image size (pixels)'},
        'galaxyfluxes':    {'type': float, 'nargs': '*', 'help': 'Galaxy fluxes [counts]'},
        'galaxyfluxmults': {'type': float, 'nargs': '+', 'help': 'Galaxy flux multipliers [counts]'},
        'galaxyradii':     {'type': float, 'nargs': '*', 'help': 'Galaxy sizes [pixels]'},
        'optlibs':    {'type': str,   'nargs': '*', 'default': 'scipy', 'help': 'Optimization libraries'},
        'algos':      {'type': str,   'nargs': '*', 'default': 'L-BFGS-B', 'help': 'Optimization algorithms'},
        'engines':    {'type': str,   'nargs': '*', 'default': 'galsim', 'help': 'Model generation engines'},
    }

    for key, value in flags.items():
        if key in options:
            default = options[key]["default"]
        else:
            default = value['default']
        if 'help' in value:
            value['help'] += ' (default: ' + str(default) + ')'
        value["default"] = default
        parser.add_argument('-' + key, **value)

    args = parser.parse_args()
    argvars = vars(args)
    optvars = ["optlibs", "algos", "engines"]
    testargs = {key: [argvars[key]] if isinstance(argvars[key], str) else argvars[key] for key in optvars}
    # TODO: Need a dict/method to get algorithm name for given optlib
    # testargs["optlibswithopts"] = {optlib: {"algo": testargs["algos"]} for optlib in testargs["optlibs"]}
    # testargs["engineswithopts"] = {engine: None for engine in testargs["engines"]}
    for optvar in optvars:
        del argvars[optvar]

    fitmodels = gettestfittingmodels(**argvars)
    psffitsall = []

    psfs = copy.deepcopy(fitmodels["modelspsf"])
    for psftype, paramspsfs in fitmodels["psfs"].items():
        for paramspsf in paramspsfs:
            psffits = testfitting(fitmodels["modelspsf"], {psftype: [paramspsf]},
                                  backgroundsbyband=fitmodels["backgroundsbyband"], engine="galsim",
                                  optlib=options["psfoptlib"]["default"], optlibopts={"algo": "neldermead"},
                                  printsteps=100)
            galfitsall = []
            psf = psfs[psftype]
            for dest, valuenew in zip(psf.getparameters(), paramspsf):
                dest.setvalue(valuenew, transformed=True)
            modelnormalize(psf)
            # Note: conveniently, the fitmodels already have the best-fit values stored
            for psfmodel in [None] + list(fitmodels["psfs"].values()):
                if psfmodel is None:
                    psffit = fitmodels["modelspsf"][psftype]
                else:
                    psffit = psfmodel
                    modelnormalize(psffit)
                for engine in testargs["engines"]:
                    galfits = testfitting(fitmodels["models"], fitmodels["galaxies"],
                                          backgroundsbyband=fitmodels["backgroundsbyband"],
                                          psf=psf, psfusemodel=True, psfname=psftype,
                                          psfmodel=psffit, psfmodelusemodel=psfmodel is not None,
                                          engine=engine,
                                          optlib=options["psfoptlib"]["default"],
                                          optlibopts={"algo": "neldermead"},
                                          printsteps=200, plot=False)
                    galfitsall.append(galfits)
            psffitsall.append((psffits,galfitsall))

    pickle.dump(psffitsall, file=os.path.expanduser("~/pyprofit.pickle.dat"))
