import argparse
import copy
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import time

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
def modelrescale(model, fluxnew=1.0):
    fluxparams = [param for param in model.getparameters() if isinstance(param, proobj.FluxParameter) and
                  not param.isfluxratio]
    fluxtotal = {}
    for param in fluxparams:
        band = param.band
        if band not in fluxtotal:
            fluxtotal[band] = 0.
        fluxtotal[band] += param.getvalue(transformed=False)
    for param in fluxparams:
        param.setvalue(fluxnew*param.getvalue(transformed=False)/fluxtotal[param.band])


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
        if psfmodel is None:
            psfmodelname = ""
    else:
        psfname = None
        psfmodelname = None

    modelinits = {
        modeltype: [param.getvalue(transformed=True)
                    for param in model.getparameters()]
        for modeltype, model in models.items()
    }

    paramsbytypecount = len(paramsbymodeltype)
    modelscount = len(models)
    # The true model
    for modeli, (modeltype, paramsoftype) in enumerate(paramsbymodeltype.items()):
        model = models[modeltype]
        paramscount = len(paramsoftype)
        # The model params
        for parami, params in enumerate(paramsoftype):
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
            print("Object params:", params)

            # The model to fit
            for modelfiti, (modeltypefit, modelfit) in enumerate(models.items()):
                logmsg = "Fitting object={:s} with model={:s}".format(modeltype, modeltypefit)
                if psfname is not None:
                    logmsg += " and PSF={:s}".format(psfname)
                if psfmodelname is not None:
                    logmsg += " and PSFmodel={:s}".format(psfmodelname)
                logmsg += " [model {}/{} object {}/{} modelfit {}/{}]".format(
                    modeli+1, paramsbytypecount, parami+1, paramscount, modelfiti+1, modelscount)
                print(logmsg)
                if modeltypefit == modeltype:
                    paramsinit = params
                else:
                    paramsinit = modelinits[modeltypefit]
                for dest, valuenew in zip(modelfit.getparameters(), paramsinit):
                    dest.setvalue(valuenew, transformed=True)
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
                try:
                    result = modeller.fit(printsteps=printsteps, printfinal=True, timing=timing)
                except Exception as e:
                    result = {"paramsbest": None, "time": None}
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
                result = {
                    "params": params,
                    "paramsbest": result["paramsbest"],
                    "modeltype": modeltype,
                    "modeltypefit": modeltypefit,
                    "psfname": psfname,
                    "psfmodelname": psfmodelname,
                    "time": result["time"],
                }
                results.append(result)
    return results


def gettestfittingmodels(
    bands="", backgrounds=options["backgrounds"]["default"],
    psfsizes=options["psfsizes"]["default"], psffluxes=options["psffluxes"]["default"],
    psfradii=options["psfradii"]["default"], psfaxrats=options["psfaxrats"]["default"],
    imagesizes=options["imagesizes"]["default"], galaxyfluxes=options["galaxyfluxes"]["default"],
    galaxyfluxmults=options["galaxyfluxmults"]["default"], galaxyradii=options["galaxyradii"]["default"],
    minimalmodels=False
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

    fluxesbyband = {i[0]: i[1] for i in zip(bands, psffluxes)}
    angles = np.repeat(90.0, bandscount)
    # Note: The ratio of radii in the double Gaussian produces a PSF with nearly the same R50 (not FWHM)
    modelspsf = {
        "gaussian:1": proutil.getmodel(fluxesbyband, "gaussian:1", psfsizes, psfradii, psfaxrats,
                                       np.repeat(90.0, bandscount)),
    }
    if not minimalmodels:
        modelspsf["gaussian:2"] = proutil.getmodel(
            fluxesbyband, "gaussian:2", psfsizes,
            np.concatenate((psfradii * 1.5, psfradii * 0.725)),
            np.concatenate((psfaxrats, psfaxrats)),
            np.concatenate((angles, angles)))
        modelspsf["moffat:1"] = proutil.getmodel(
            fluxesbyband, "moffat:1", psfsizes, psfradii, psfaxrats,
            angles, slopes=np.repeat(2.5, bandscount))

    psfs = {model: [] for model in modelspsf}

    # TODO: Is this really easier than just changing the value(s) of the relevant parameters?
    # It's definitely not faster...
    for angle in np.linspace(0., 45., 2 + 2*(not minimalmodels)):
        angles = np.repeat(angle, bandscount)
        model = "gaussian:1"
        psfs[model].append(getparamvalues(proutil.getmodel(
            fluxesbyband, model, psfsizes, psfradii, psfaxrats, angles, nexposures=0)))
        model = "gaussian:2"
        if model in modelspsf:
            psfs[model].append(getparamvalues(proutil.getmodel(
                fluxesbyband, model, psfsizes, np.concatenate((psfradii*2., psfradii*0.5)),
                np.concatenate((psfaxrats, psfaxrats)), np.concatenate((angles, angles)), nexposures=0)))
        # single Moffat with nominal 2.5 slope
        model = "moffat:1"
        if model in modelspsf:
            psfs[model].append(getparamvalues(proutil.getmodel(
                fluxesbyband, model, psfsizes, psfradii, psfaxrats, angles, slopes=np.repeat(2.5, bandscount),
                nexposures=0)))

    galaxyfluxes = np.array(galaxyfluxes)
    galaxyradii = np.array(galaxyradii)
    galaxyfluxmults = np.array(galaxyfluxmults)

    fluxesbyband = {i[0]: i[1] for i in zip(bands, galaxyfluxes)}
    models = {
        "sersic:1": proutil.getmodel(fluxesbyband, "sersic:1", imagesizes, np.repeat(5., bandscount),
                                     np.repeat(0.5, bandscount), angles, slopes=np.repeat(2., bandscount)),
    }
    if not minimalmodels:
        models["expser"] = proutil.getmodel(
            fluxesbyband, "sersic:2", imagesizes,
            np.concatenate((np.repeat(5., bandscount),
                            np.repeat(2., bandscount))),
            np.concatenate((np.repeat(0.5, bandscount),
                            np.repeat(0.9, bandscount))),
            np.concatenate((angles, angles)),
            slopes=np.concatenate((np.repeat(1., bandscount),
                                   np.repeat(4., bandscount))),
            fluxfracs=np.concatenate((np.repeat(0.95, bandscount),
                                   np.repeat(0.05, bandscount))),
        )

    for component in models["expser"].sources[0].modelphotometric.components:
        component.parameters["nser"].fixed = True

    galaxies = {model: [] for model in models}

    # See Cortese et al. 2016 (or modify to suit your taste)
    diskheightratio = 0.2
    if minimalmodels:
        bulgeheightratios = np.array([0.5, 1.0])
        cosis = np.linspace(0., 1., 3)
    else:
        bulgeheightratios = np.array([0.4, 0.7, 1.0])
        cosis = np.linspace(0., 1., 5)
    sersicnbulges = [4.] if minimalmodels else [2., 4.]
    bulgefracs = [0.5] if minimalmodels else [0.2, 0.5, 0.8]
    sersicndisks = [1.] if minimalmodels else [0.5, 1.0, 2.]
    angles = np.linspace(0., 45., 2 + 2*(not minimalmodels))
    if minimalmodels:
        angles = np.concatenate(angles, angles+90.)
    for fluxmult in galaxyfluxmults:
        fluxesbyband1 = {i[0]: fluxmult*i[1] for i in zip(bands, galaxyfluxes)}
        for radius in galaxyradii:
            radii = np.repeat(radius, bandscount)
            for angle in angles:
                anglesinit = np.repeat(angle, bandscount)
                model = "sersic:1"
                # See Cortese et al. 2016 eqn. (5)
                axrats = np.sqrt((cosis * (1-diskheightratio**2))**2 + diskheightratio**2)
                if "sersic:1" in models:
                    # Pure exponential disk galaxy
                    for axrat in axrats:
                        galaxies[model].append(getparamvalues(proutil.getmodel(
                            fluxesbyband1, "sersic:1", imagesizes, radii, np.repeat(axrat, bandscount),
                            anglesinit, slopes=np.repeat(1., bandscount), nexposures=0)))
                    if not minimalmodels:
                        # Sersic bulge (elliptical-like galaxy)
                        axrats = set()
                        for cosi in cosis:
                            for heightratio in bulgeheightratios:
                                axrats.add(np.sqrt((cosi * (1.-heightratio**2))**2 + heightratio**2))
                        for axrat in axrats:
                            for sersicn in sersicnbulges:
                                galaxies[model].append(getparamvalues(proutil.getmodel(
                                    fluxesbyband1, "sersic:1", imagesizes, radii, np.repeat(axrat, bandscount),
                                    anglesinit, slopes=np.repeat(sersicn, bandscount), nexposures=0)))
                if "expser" in models and not minimalmodels:
                    model = "expser"
                    axrats = np.sqrt((cosis * (1 - diskheightratio ** 2)) ** 2 + diskheightratio ** 2)
                    for axrat in axrats:
                        for bulgefrac in bulgefracs:
                            for sersicndisk in sersicndisks:
                                for sersicnbulge in sersicnbulges:
                                    galaxies[model].append(getparamvalues(proutil.getmodel(
                                        fluxesbyband1, "sersic:2", imagesizes,
                                        np.concatenate((np.repeat(1.2*radius, bandscount),
                                                        np.repeat(0.4*radius, bandscount))),
                                        np.concatenate((np.repeat(axrat, bandscount),
                                                        np.repeat(1., bandscount))),
                                        np.concatenate((angles, angles)),
                                        slopes=np.concatenate((
                                            np.repeat(sersicndisk, bandscount),
                                            np.repeat(sersicnbulge, bandscount))),
                                        fluxfracs=np.concatenate((
                                            np.repeat(1.-bulgefrac, bandscount),
                                            np.repeat(1.-bulgefrac, bandscount))),
                                        nexposures=0)))


                # exp + dev
                # double Sersic

    print("Returning {} models and {} PSFs to test".format(
        sum(len(galaxies[model]) for model in galaxies),
        sum(len(psfs[model]) for model in psfs)
    ))

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
        'plot':       {'type': proutil.str2bool, 'default': False, 'help': 'Toggle plotting of final fits'},
        'fileout':    {'type': str,   'nargs': '?', 'default': None, 'help': 'File prefix to output results'},
        'seed':       {'type': int,   'nargs': '?', 'default': 1, 'help': 'Numpy random seed'}
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
    fileout = args.fileout
    seed = args.seed
    optvars = ["optlibs", "algos", "engines", "plot"]
    testargs = {key: [argvars[key]] if isinstance(argvars[key], str) else argvars[key] for key in optvars}
    # TODO: Need a dict/method to get algorithm name for given optlib
    # testargs["optlibswithopts"] = {optlib: {"algo": testargs["algos"]} for optlib in testargs["optlibs"]}
    # testargs["engineswithopts"] = {engine: None for engine in testargs["engines"]}
    for optvar in optvars:
        del argvars[optvar]
    for var in ['fileout', 'seed']:
        del argvars[var]

    fitmodels = gettestfittingmodels(**argvars)

    write = fileout is not None

    psfs = copy.deepcopy(fitmodels["modelspsf"])
    # The actual PSF
    for psftype, paramspsfs in fitmodels["psfs"].items():
        for psfi, paramspsf in enumerate(paramspsfs):
            np.random.seed(seed)
            psffits = testfitting(fitmodels["modelspsf"], {psftype: [paramspsf]},
                                  backgroundsbyband=fitmodels["backgroundsbyband"], engine="galsim",
                                  optlib=options["psfoptlib"]["default"], optlibopts={"algo": "neldermead"},
                                  printsteps=None)
            seed = seed + 1
            galfitsall = {}
            psf = psfs[psftype]
            for dest, valuenew in zip(psf.getparameters(), paramspsf):
                dest.setvalue(valuenew, transformed=True)
            modelrescale(psf)
            # The PSF to use
            psfmodels = [None] + psffits
            for psfmodel in psfmodels:
                np.random.seed(seed)
                if psfmodel is None:
                    psffit = fitmodels["modelspsf"][psftype]
                    psfmodelname = ""
                else:
                    psffit = fitmodels["modelspsf"][psfmodel["modeltype"]]
                    psfmodelname = psfmodel["modeltype"]
                    # Note: conveniently, the fitmodels already have the best-fit values stored
                    #for dest, valuenew in zip(psffit.getparameters(fixed=False), psfmodel["fitresult"][0]):
                    #    dest.setvalue(valuenew, transformed=True)
                    modelrescale(psffit)
                for engine in testargs["engines"]:
                    galfits = testfitting(fitmodels["models"], fitmodels["galaxies"],
                                          backgroundsbyband=fitmodels["backgroundsbyband"],
                                          psf=psf, psfusemodel=True, psfname=psftype,
                                          psfmodel=psffit, psfmodelname=psfmodelname,
                                          psfmodelusemodel=psfmodel is not None,
                                          engine=engine,
                                          optlib=options["psfoptlib"]["default"],
                                          optlibopts={"algo": "neldermead"},
                                          plot=testargs["plot"], printsteps=None,
                                          )
                    if write:
                        galfitsall[engine] = galfits
                if write:
                    data = {
                        "fitmodels": fitmodels,
                        "galfits": galfitsall,
                        "psfmodel": psfmodel,
                        "psfname": psftype,
                        "psfmodelname": psfmodelname,
                        "seed": seed,
                    }

                    import _pickle as pickle
                    filepsf = "_".join(fileout, psftype, psfi, psfmodelname) + ".dat"
                    with open(os.path.expanduser(filepsf), 'wb') as f:
                        pickle.dump(data, f)
                seed = seed + 1
