import argparse
import astropy as ap
from collections import OrderedDict
import copy
import galsim as gs
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.optimize as spopt
import traceback
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


# Fairly standard moment of inertia estimate of ellipse orientation and size
# TODO: compare with galsim's convenient calculateHLR/FWHM
def getellipseestimate(img, denoise=True):
    imgmeas = proutil.absconservetotal(np.copy(img)) if denoise else img
    y, x = np.nonzero(imgmeas)
    flux = imgmeas[y, x]
    y = y - imgmeas.shape[0]/2.
    x = x - imgmeas.shape[1]/2.
    inertia = np.zeros((2,2))
    inertia[0, 0] = np.sum(flux*x**2)
    inertia[0, 1] = np.sum(flux*x*y)
    inertia[1, 0] = inertia[0, 1]
    inertia[1, 1] = np.sum(flux*y**2)
    evals, evecs = np.linalg.eig(inertia)
    idxevalmax = np.argmax(evals)
    axrat = evals[1-idxevalmax]/evals[idxevalmax]
    ang = np.degrees(np.arctan2(evecs[1, idxevalmax], evecs[0, idxevalmax])) - 90
    if ang < 0:
        ang += 360
    return axrat, ang, np.sqrt(evals[idxevalmax]/np.sum(flux))


def getchisqred(chis):
    chisum = 0
    chicount = 0
    for chivals in chis:
        chisum += np.sum(chivals**2)
        chicount += len(chivals)**2
    return chisum/chicount


def fitmodel(model, modeller=None, modellib=None, modellibopts=None, printfinal=True, printsteps=100,
             plot=False, modelname=None, figure=None, title=None,
             axes=None, figurerow=None, modelnameappendparams=None):
    if modeller is None:
        modeller = proobj.Modeller(model=model, modellib=modellib, modellibopts=modellibopts)
    fit = modeller.fit(printfinal=printfinal, printsteps=printsteps)
    # Conveniently sets the parameters to the right values too
    if plot:
        modeldesc = modelname
        if modelnameappendparams is not None:
            for string, param in modelnameappendparams:
                modeldesc += string.format(param.getvalue(transformed=False))
    else:
        modeldesc = None
    _, _, chis = model.evaluate(params=fit["paramsbest"], plot=plot, modelname=modeldesc, figure=figure,
                                axes=axes, figurerow=figurerow)
    if plot and title is not None:
        plt.suptitle(title)
    fit["chisqred"] = getchisqred(chis)
    fit["paramsbestall"] = [param.getvalue(transformed=True) for param in model.getparameters()]
    return fit, modeller


def setexposure(model, band, image=None, sigmainverse=None, psf=None, mask=None, factorsigma=1):
    exposure = model.data.exposures[band][0]
    exposure.image = image
    if psf is None and image is not None and sigmainverse is None:
        sigmaimg = np.sqrt(np.var(image))
        exposure.sigmainverse = 1.0/(factorsigma*sigmaimg)
    else:
        exposure.sigmainverse = sigmainverse
    exposure.psf = psf
    exposure.mask = mask
    return model


def getpsfmodel(engine, engineopts, numcomps, band, psfmodel, psfimage, factorsigma=1):
    if engine == "galsim":
        engineopts = {"gsparams": gs.GSParams(kvalue_accuracy=1e-4,
                                              integration_relerr=1e-4,
                                              integration_abserr=1e-6)}
    model = proutil.getmodel({band: 1}, psfmodel, np.flip(psfimage.shape, axis=0),
                             8.0 * 10 ** ((np.arange(numcomps) - numcomps / 2) / numcomps),
                             np.repeat(0.8, numcomps),
                             np.linspace(start=0, stop=180, num=numcomps + 2)[1:(numcomps + 1)],
                             engine=engine, engineopts=engineopts)
    for param in model.getparameters(fixed=False):
        if isinstance(param, proobj.FluxParameter) and not param.isfluxratio:
            param.fixed = True
    setexposure(model, band, image=psfimage, factorsigma=factorsigma)
    return model


def fitgalaxy(img, psf, sigmainverse, band, mask=None, modellib=None, algo=None, plot=False, psfmodel=None,
              dofullbulgedisk=False, factorsigmapsf=1, name=None,
              modelsinitfrom=dict(
                  #exp=["multiexp"],
                  #dev=["multidev"],
                  ser=["gauss", "exp", "dev"],
              )):
    """

    :param img: ndarray; 2D Image
    :param psf: ndarray; 2D PSF Image
    :param sigmainverse: ndarray; 2D Inverse sigma image ndarray
    :param band: string; Filter/passband name
    :param mask: ndarray; 2D Inverse mask image (1=include, 0=omit)
    :param modellib: string; Model fitting library
    :param algo: string; Fitting algorithm
    :param plot: bool; make plots?
    :param psfmodel: string; PSF model description e.g. "gaussian:2"
    :param dofullbulgedisk: bool; Do full bulge-disk decomposition vs just CModel
    :param factorsigmapsf: float; Factor to multiply the PSF sigma image by
    :param modelsinitfrom: dict; key=model name: value=array of models to initialize from (selects best fit)

    :return: modelinfos, models, psfmodels: tuple of complicated structures:
        modelinfos: dict; key=model name: value=dict; TBD
        models: dict; key=engine name: value=dict(key=model type: value=proobj.Model of that type)
        psfmodels: dict: TBD
    """
    axrat, ang, re = getellipseestimate(img.array)
    engines = {
        "galsim": {"gsparams": gs.GSParams(kvalue_accuracy=1e-2, integration_relerr=1e-2,
                                           integration_abserr=1e-3, maximum_fft_size=16384)}
    }
    title = name if plot else None
    psfmodels = {}
    # Fit the PSF
    if psfmodel is not None:
        numcomps = np.int(psfmodel.split(":")[1])
        for engine, engineopts in engines.items():
            model = getpsfmodel(engine, engineopts, numcomps, band, psfmodel, psf.image.array,
                                factorsigma=factorsigmapsf)
            modeller = proobj.Modeller(model=model, modellib="pygmo", modellibopts={"algo": "neldermead"})
            fit = fitmodel(model, modeller, printfinal=True, printsteps=100, plot=plot,
                           title=title, modelname=psfmodel + " PSF")
            psfmodels[engine] = {"model": model, "fit": fit}

    npiximg = np.flip(img.array.shape, axis=0)
    flux = np.sum(img.array[mask] if mask is not None else img.array)
    models = {
        engine: {
            "sersic": proutil.getmodel({band: flux}, "gaussian:1", npiximg,
                                       [re], [axrat], [ang], engine=engine, engineopts=engineopts),
            "multigaussian": proutil.getmodel({band: flux}, "multigaussiansersic:1", npiximg,
                                              [re], [axrat], [ang], slopes=[1.0], engine=engine,
                                              engineopts=engineopts),
        } for engine, engineopts in engines.items()
    }
    for engine, modelsbytype in models.items():
        for modeltype, model in modelsbytype.items():
            init = [param.getvalue(transformed=True) for param in model.getparameters(fixed=False)]
            fixedinit = [param.fixed for param in model.getparameters()]
            if psfmodel is None:
                psfexposure = proobj.PSF(band=band, engine="galsim", image=psf)
            else:
                psfexposure = proobj.PSF(band=band, engine="galsim",
                                         model=psfmodels[engine]["model"].sources[0])
            setexposure(model, band, img.array, np.zeros(img.array.shape) + sigmainverse, psfexposure,
                        mask=mask)
            modeller = proobj.Modeller(model=model, modellib="scipy", modellibopts={"algo": "L-BFGS-B"})
            models[engine][modeltype] = {
                "fixedinit": fixedinit,
                "init": init,
                "model": model,
                "modeller": modeller,
            }

    modelssersic = [
        ("gauss", 0.5),
        ("multiexp", 1.0),
        ("exp", 1.0),
        ("multidev", 4.0),
        ("dev", 4.0),
        ("ser", 2.0),
        # ,
        #,
    ]
    modelinfos = OrderedDict()
    for model in modelssersic:
        modelinfos[model[0]] = {"nser": model[1]}
    errors = []
    for model, modelsfrom in modelsinitfrom.items():
        modelsinit = [modelfrom in modelsinitfrom for modelfrom in modelsfrom]
        if any(modelsinit):
            errors.append("Model {} initializes from models {} which also initialize from other models; "
                          "resolving priority chains is not implemented yet".format(
                             model, ",".join(np.array(modelsfrom)[modelsinit])))
        modelinfos.move_to_end(model)
    if errors:
        # TODO: Fix this
        print("Warnings: " + " && ".join(errors))
        #raise RuntimeError("Errors: " + " && ".join(errors))

    remax = np.sqrt(np.sum((npiximg/2.)**2))

    for modelname in modelinfos:
        modelinfos[modelname]["fits"] = {}
    for engine in engines:
        if plot:
            # Will be used below
            figurerow = len(modelinfos)
            figure, axes = plt.subplots(nrows=figurerow+2, ncols=5)
            plt.suptitle(title + " {} model".format(engine))
        else:
            figurerow = None
            figure = None
            axes = None
        for modelidx, modelname in enumerate(modelinfos):
            print("Fitting model {:s} using engine {:s}".format(modelname, engine))
            model = models[engine]
            nser = modelinfos[modelname]["nser"]
            if modelname.startswith("multi"):
                model = model["multigaussian"]
            else:
                model = model["sersic"]
            fits = []
            # Fit the position and re first
            fixedinit = model["fixedinit"]
            init = model["init"]
            modeller = model["modeller"]
            model = model["model"]
            toinitfromother = modelname in modelsinitfrom
            if toinitfromother:
                # Check chisqred of reference models to find best model
                modelchisqreds = {modelfit: modelinfos[modelfit]["fits"][engine][-1]["chisqred"]
                                  for modelfit in modelsinitfrom[modelname]}
                modelbest = min(modelchisqreds, key=modelchisqreds.get)
                print("Model {} initializing from best-fit model={} chisqred={:.2e}".format(
                    modelname, modelbest, modelchisqreds[modelbest]
                ))
                init = modelinfos[modelbest]["fits"][engine][-1]["paramsbestall"]
            for value, param in zip(init, model.getparameters(fixed=toinitfromother)):
                param.setvalue(value, transformed=True)
                if param.name == "re":
                    param.limits = copy.deepcopy(param.limits)
                    param.limits.upper = (param.transform.transform(remax) if
                                          param.limits.transformed else remax)
                    if not toinitfromother and (modelname == "dev" or modelname == "multidev"):
                        re = param.getvalue(transformed=False)
                        param.setvalue(re*(1 + 0.25*((re < 1) + (re < 10))), transformed=False)
                elif not (param.name == "cenx" or param.name == "ceny" or
                          (param.name == "flux" and modelname == "dev")):
                    param.fixed = True
            for param in model.getparameters(free=False):
                if param.name == "nser":
                    if toinitfromother:
                        param.fixed = False
                    else:
                        param.setvalue(nser, transformed=False)
            modeller.modellib = "scipy"
            modeller.modellibopts["algo"] = "L-BFGS-B"
            fit1, _ = fitmodel(model, modeller, printfinal=True, printsteps=100)
            fits.append(fit1)
            # Free fixed parameters as needed
            for param, fixed in zip(model.getparameters(), fixedinit):
                if param.name == "nser" and modelname == "ser":
                    param.fixed = False
                    paramser = param
                else:
                    param.fixed = fixed
            #fit2, _ = fitmodel(model, modeller, printfinal=True, printsteps=100)
            #fits.append(fit2)
            modeller.modellib = "pygmo"
            modeller.modellibopts["algo"] = "neldermead"
            fit3, _ = fitmodel(model, modeller, printfinal=True, printsteps=100,
                               plot=plot, figure=figure, axes=axes, figurerow=modelidx,
                               modelname=modelname, modelnameappendparams=[(" n={:.2f}", paramser)] if
                modelname == "ser" else [])
            fits.append(fit3)
            modelinfos[modelname]["fits"][engine] = fits
    paramsinit = ["re", "axrat", "ang"]
    fluxratmax = 0.99
    fluxratmin = 0.01
    for modelname in ["cmodel", "devexp", "serexp", "serser"]:
        modelinfos[modelname] = {"fits": {}}
    for engine, engineopts in engines.items():
        valuesinit = {paramname: [] for paramname in paramsinit}
        for modelname in ["multidev", "exp"]:
            fit = modelinfos[modelname]["fits"][engine][1]
            paramsbest = dict(zip(fit["paramnames"], fit["paramsbest"]))
            for paramname in paramsinit:
                valuesinit[paramname].append(paramsbest[paramname])
            if modelname == "exp":
                offsetxy = (np.array([paramsbest["cenx"], paramsbest["ceny"]]) -
                            np.flip(img.array.shape, axis=0)/2)

        # TODO: We're assuming a log10 transform here but should check it first
        # ... or allow passing untransformed slopes, or grab the value from the old models?
        modeltype = "multigaussiansersic:1,sersic:1"
        model = proutil.getmodel({band: flux}, modeltype, npiximg,
                                 valuesinit["re"], valuesinit["axrat"], valuesinit["ang"],
                                 offsetxy=offsetxy, engine=engine, engineopts=engineopts,
                                 istransformedvalues=True, slopes=[np.log10(4.), 0])
        modelname = "cmodel"
        # TODO: write function to do this since it's repeated
        exposure = model.data.exposures[band][0]
        exposure.image = img.array
        if psfmodel is None:
            exposure.psf = proobj.PSF(band=band, engine="galsim", image=psf)
        else:
            exposure.psf = proobj.PSF(band=band, engine="galsim",
                                      model=psfmodels[engine]["model"].sources[0])
        exposure.sigmainverse = np.zeros(img.array.shape) + sigmainverse
        for param in model.getparameters(fixed=False):
            if param.name == "flux":
                if param.isfluxratio:
                    param.limits = copy.copy(param.limits)
                    param.limits.lower = (param.transform.transform(fluxratmin) if
                                          param.limits.transformed else fluxratmin)
                    param.limits.upper = (param.transform.transform(fluxratmax) if
                                          param.limits.transformed else fluxratmax)
                    if not param.fixed:
                        param.setvalue(0.2, transformed=False)
            else:
                param.fixed = True
        # A clever way to append flux fractions for components since the plot should show the final B/T
        appendparams = [(" f={:.2f}", param) for param in model.getparameters(fixed=False) if
                        isinstance(param, proobj.FluxParameter) and param.isfluxratio]
        # fit
        fit, modeller = fitmodel(
            model, modellib="pygmo", modellibopts={"algo": "neldermead"}, plot=plot, title=title,
            modelname=modelname, figure=figure, axes=axes, figurerow=figurerow,
            modelnameappendparams=appendparams
        )
        modelinfos[modelname]["fits"][engine] = [fit]
        # fit again with all but nser free, and with a lower limit on the bulge axis ratio
        modelname = "devexp"
        axratmin = 0.5
        # If fracdev < 0.5 and rebulge > redisk, reset the bulge size to half of the disk size
        # Could also use nser instead
        for param in model.getparameters(fixed=False):
            if param.name == "flux" and param.isfluxratio:
                if param.getvalue(transformed=False) < 0.5 and valuesinit["re"][0] > valuesinit["re"][1]:
                    rebulge = valuesinit["re"][1]
                else:
                    rebulge = valuesinit["re"][0]
        hassetbulge = {var: False for var in ["axrat", "re"]}
        for param in model.getparameters(fixed=True):
            if param.name in paramsinit + ["cenx", "ceny"]:
                param.fixed = False
            if param.name == "re":
                param.limits = copy.deepcopy(param.limits)
                remaxcomp = remax
                if not hassetbulge[param.name]:
                    param.setvalue(rebulge, transformed=True)
                    remaxcomp = param.getvalue(transformed=False)
                    param.setvalue(remaxcomp/2, transformed=False)
                    hassetbulge[param.name] = True
                param.limits.upper = (param.transform.transform(remaxcomp) if
                    param.limits.transformed else remaxcomp)
            elif param.name == "axrat" and not hassetbulge[param.name]:
                param.limits = copy.deepcopy(param.limits)
                param.limits.lower = (param.transform.transform(axratmin) if
                                      param.limits.transformed else axratmin)
                if param.getvalue(transformed=False) < axratmin:
                    param.setvalue(axratmin, transformed=False)
                hassetbulge[param.name] = True
        fit1, _ = fitmodel(model, modeller, printfinal=True, printsteps=100)
        if hassetbulge["re"]:
            # Run again without the limit on rebulge
            for param, value in zip(model.getparameters(fixed=False), fit1["paramsbest"]):
                if param.name == "re":
                    param.limits.upper = (param.transform.transform(remaxcomp) if
                        param.limits.transformed else remaxcomp)
                param.setvalue(value, transformed=True)
        # It won't hurt to restart the fit in both cases
        if plot:
            figurerow += 1
        appendparams = [(" f={:.2f}", param) for param in model.getparameters(fixed=False) if
                        isinstance(param, proobj.FluxParameter) and param.isfluxratio]
        fit2, _ = fitmodel(
            model, modeller,
            plot=plot, title=title, modelname=modelname,
            figure=figure, axes=axes, figurerow=figurerow, modelnameappendparams=appendparams
        )
        modelinfos[modelname]["fits"][engine] = [fit1, fit2]
        if dofullbulgedisk:
            # fit again with free bulge n
            countnser = 0
            for param in model.getparameters(fixed=True):
                if param.name == "nser":
                    if countnser == 0:
                        param.fixed = False
                    countnser += 1
            fit = modeller.fit(printfinal=True)
            modelinfos["serexp"]["fits"][engine] = fit
            model.getlikelihood(params=fit["paramsbest"], plot=plot)
            # fit one last time with everything free
            countnser = 0
            for param in model.getparameters(fixed=True):
                if param.name == "nser":
                    if countnser == 1:
                        param.fixed = False
                    countnser += 1
            fit = modeller.fit(printfinal=True)
            model.getlikelihood(params=fit["paramsbest"], plot=plot)
            modelinfos["serser"]["fits"][engine] = [fit]
        models[engine][modeltype] = dict(model=model, modeller=modeller)
    return modelinfos, models, psfmodels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyProFit HST COSMOS galaxy modelling test')

    signature = inspect.signature(fitgalaxy)
    defaults = {
        k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty
    }

    flags = {
        'catalogpath': {'type': str, 'nargs': '?', 'default': None, 'help': 'File prefix to output results'},
        'catalogfile': {'type': str, 'nargs': '?', 'default': None, 'help': 'File prefix to output results'},
        'fithst':      {'type': proutil.str2bool, 'default': False, 'help': 'Fit HSC I band image'},
        'fithsc':      {'type': proutil.str2bool, 'default': False, 'help': 'Fit HST F814W image'},
        'fithst2hsc':  {'type': proutil.str2bool, 'default': False, 'help': 'Fit HST F814W image convolved '
                                                                            'to HSC seeing'},
        'psfmodel':    {'type': str,   'default': None, 'help': 'PSF model'},
        'psfsizes':    {'type': int,   'nargs': '*', 'help': 'PSF image sizes (pixels)'},
        'psffluxes':   {'type': float, 'nargs': '*', 'help': 'PSF fluxes [counts]'},
        'psfradii':    {'type': float, 'nargs': '*', 'help': 'PSF D_50 (=FWHM for Gaussians only) [pixels]'},
        'psfaxrats':   {'type': float, 'nargs': '*', 'help': 'PSF ellipticities [fraction]'},
#        'imagesizes': {'type': int,   'nargs': '*', 'help': 'Galaxy image size (pixels)'},
#        'galaxyfluxes':    {'type': float, 'nargs': '*', 'help': 'Galaxy fluxes [counts]'},
#        'galaxyfluxmults': {'type': float, 'nargs': '+', 'help': 'Galaxy flux multipliers [counts]'},
#        'galaxyradii':     {'type': float, 'nargs': '*', 'help': 'Galaxy sizes [pixels]'},
        'modellib':    {'type': str,   'nargs': '?', 'default': 'pygmo', 'help': 'Optimization libraries'},
        'algo':        {'type': str,   'nargs': '?', 'default': 'neldermead',
                       'help': 'Optimization algorithms'},
#        'engines':    {'type': str,   'nargs': '*', 'default': 'galsim', 'help': 'Model generation engines'},
        'plot':        {'type': proutil.str2bool, 'default': False, 'help': 'Toggle plotting of final fits'},
        'fileout':    {'type': str,   'nargs': '?', 'default': None, 'help': 'File prefix to output results'},
        'indices':     {'type': str, 'nargs': '*', 'default': None, 'help': 'Galaxy catalog index'},
#        'seed':       {'type': int,   'nargs': '?', 'default': 1, 'help': 'Numpy random seed'}
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
    args.catalogpath = os.path.expanduser(args.catalogpath)

    try:
        rgcat = gs.RealGalaxyCatalog(args.catalogfile, dir=args.catalogpath)
    except Exception as e:
        print("Failed to load RealGalaxyCatalog {} in directory {}".format(
            args.catalogfile, args.catalogpath))
        print("Exception:", e)
        raise e
    try:
        ccat = gs.COSMOSCatalog(args.catalogfile, dir=args.catalogpath)
    except Exception as e:
        print("Failed to load COSMOSCatalog {} in directory {}".format(
            args.catalogfile, args.catalogpath))
        print("Not using COSMOSCatalog")
        ccat = None

    if args.fileout is not None and os.path.isfile(args.fileout):
        with open(os.path.expanduser(args.fileout), 'rb') as f:
            data = pickle.load(f)
    else:
        data = {}

    if args.plot:
        mpl.rcParams['image.origin'] = 'lower'

    rgcfits = ap.io.fits.open(os.path.join(args.catalogpath, args.catalogfile))[1].data
    srcs = ["hst"] if args.fithst else []
    if args.fithsc or args.fithst2hsc:
        from modelling_research import make_cutout
        import lsst.afw.geom as geom
        import lsst.daf.persistence as dafPersist
        from lsst.meas.base.measurementInvestigationLib import rebuildNoiseReplacer
        from lsst.afw.table import SourceTable
        from lsst.meas.base.measurementInvestigationLib import makeRerunCatalog
        from lsst.meas.base import (SingleFrameMeasurementConfig,
                                    SingleFrameMeasurementTask)
        tract = 9813
        butler = dafPersist.Butler("/datasets/hsc/repo/rerun/RC/w_2018_30/DM-15120/")
        dataId = {"tract": tract}
        skymap = butler.get("deepCoadd_skyMap", dataId=dataId)
    if args.fithsc:
        srcs += ["hsc"]
    if args.fithst2hsc:
        srcs += ["hst2hsc"]

    nfit = 0
    for index in args.indices:
        idrange = [np.int(x) for x in index.split(",")]
        for id in range(idrange[0], idrange[0+(len(idrange)>1)]+1):
            print("Fitting COSMOS galaxy with ID: {}".format(id))
            try:
                radec = rgcfits[id][1:3]
                imghst = rgcat.getGalImage(id)
                fluxhst = rgcat.stamp_flux[id]
                scalehst = rgcfits[id]['PIXEL_SCALE']
                bandhst = rgcat.band[id]
                psfhst = rgcat.getPSF(id)
                if "hsc" in srcs or "hst2hsc" in srcs:
                    # Get the HSC dataRef
                    spherePoint = geom.SpherePoint(radec[0], radec[1], geom.degrees)
                    patch = skymap[tract].findPatch(spherePoint).getIndex()
                    patch = ",".join([str(x) for x in patch])
                    dataId2 = {"tract": 9813, "patch": patch, "filter": "HSC-I"}
                    dataRef = butler.dataRef("deepCoadd", dataId=dataId2)
                    # Get the coadd
                    exposure = dataRef.get("deepCoadd_calexp")
                    scalehsc = exposure.getWcs().getPixelScale().asArcseconds()
                    # Get the measurements
                    measCat = dataRef.get("deepCoadd_meas")
                    # Get and verify match
                    distsq = ((radec[0] - np.degrees(measCat["coord_ra"])) ** 2 +
                              (radec[1] - np.degrees(measCat["coord_dec"])) ** 2)
                    row = np.int(np.argmin(distsq))
                    idHsc = measCat["id"][row]
                    dist = np.sqrt(distsq[row])*3600
                    print('Source distance={:.2e}"'.format(dist))
                    # TODO: Threshold distance?
                    if dist > 1:
                        raise RuntimeError("Nearest HSC source at distance {:.3e}>1; aborting".format(dist))
                    # Determine the HSC cutout size (larger than HST due to bigger PSF)
                    sizeCutout = np.int(4 + np.ceil(np.max(imghst.array.shape) * scalehst / scalehsc))
                    sizeCutout += np.int(sizeCutout % 2)
                for src in srcs:
                    bands = [rgcat.band[id] if src == "hst" else "HSC-I"]
                    for band in bands:
                        if src == "hst":
                            img = imghst
                            psf = psfhst
                            sigmainverse = np.power(rgcat.getNoiseProperties(id)[2], -0.5)
                        else:
                            psf = exposure.getPsf()
                            scalehscpsf = psf.getWcs(0).getPixelScale().asArcseconds()
                            imgpsf = psf.computeImage().array
                            imgpsfgs = gs.InterpolatedImage(gs.Image(imgpsf, scale=scalehscpsf))
                            useNoiseReplacer = True
                            if useNoiseReplacer:
                                noiseReplacer = rebuildNoiseReplacer(exposure, measCat)
                                noiseReplacer.insertSource(idHsc)
                            cutouthsc = make_cutout.make_cutout_lsst(
                                spherePoint, exposure, size=np.floor_divide(sizeCutout, 2))
                            idshsc = cutouthsc[4]
                            var = exposure.getMaskedImage().getVariance().array[
                                  idshsc[3]: idshsc[2], idshsc[1]: idshsc[0]]
                            if src == "hst2hsc":
                                # The COSMOS GalSim catalog is in the original HST frame, which is rotated by
                                # 10-12 degrees from RA/Dec axes; fit for this
                                # Use Sophie's code to make our own cutout for comparison to the catalog
                                cutouthst = make_cutout.cutout_HST(
                                    radec[0], radec[1], width=np.ceil(sizeCutout * scalehsc),
                                    return_data=True)
                                imghstrot = gs.Image(cutouthst[0][0][1].data, scale=0.03)
                                # 0.03" scale: check cutouthst[0][0][1].header["CD1_1"] and ["CD2_2"]

                                def getoffsetdiff(x, returnimg=False):
                                    img = gs.InterpolatedImage(imghstrot).rotate(-x[0] * gs.radians).shift(
                                        x[1], x[2]).drawImage(
                                        nx=imghst.array.shape[1], ny=imghst.array.shape[0],
                                        scale=imghst.scale)
                                    if returnimg:
                                        return img
                                    return np.sum(np.abs(img.array - imghst.array))

                                result = spopt.minimize(getoffsetdiff, [-np.pi/2, 0, 0], method="Nelder-Mead")
                                result2 = spopt.minimize(getoffsetdiff, result.x + [np.pi, 0, 0],
                                                         method="Nelder-Mead")
                                [anglehst, xoff, yoff] = (result if result.fun < result2.fun else result2).x

                                if args.plot:
                                    fig, ax = plt.subplots(nrows=2, ncols=2)
                                    ax[0, 0].imshow(np.log10(imghstrot.array))
                                    ax[0, 0].set_title("HST rotated")
                                    ax[1, 0].imshow(np.log10(imghst.array))
                                    ax[1, 0].set_title("COSMOS GalSim")
                                    ax[0, 1].imshow(np.log10(getoffsetdiff([anglehst, xoff, yoff],
                                                                        returnimg=True).array))
                                    ax[0, 1].set_title("HST re-rotated+shifted")

                                realGalaxy = ccat.makeGalaxy(index=id, gal_type="real")
                                mask = None
                                # My best attempt at matching is to shift and flux scale the HST2HSC image
                                # until it's as close as possible to the HSC image
                                def getoffsetchisq(x, returnimg=False):
                                    img = gs.Convolve(
                                        gs.InterpolatedImage(imghst*10**x[0]).rotate(anglehst * gs.radians),
                                        imgpsfgs).shift(x[1], x[2]).drawImage(
                                        nx=sizeCutout, ny=sizeCutout, scale=scalehsc)
                                    chisq = np.sum((img.array-cutouthsc[0])**2/var)
                                    if returnimg:
                                        return img
                                    return chisq

                                scalefluxhst2hsc = np.sum(cutouthsc[0]) / np.sum(imghst.array)
                                result = spopt.minimize(getoffsetchisq,
                                                        [np.log10(scalefluxhst2hsc), 0, 0],
                                                        method="Nelder-Mead")

                                if args.plot:
                                    ax[1, 1].imshow(np.log10(
                                        gs.InterpolatedImage(imghst*10**result.x[0]).rotate(
                                            anglehst * gs.radians).shift(-xoff, -yoff).drawImage(
                                        nx=imghst.array.shape[1], ny=imghst.array.shape[0],
                                        scale=imghst.scale).array))
                                    ax[1, 1].set_title("COSMOS GalSim rotated+shifted")

                                # Assuming that these images match, add HSC noise back in
                                img = getoffsetchisq(result.x, returnimg=True)
                                if args.plot:
                                    fig2, ax2 = plt.subplots(nrows=2, ncols=3)
                                    ax2[0, 0].imshow(np.log10(cutouthsc[0]))
                                    ax2[0, 0].set_title("HSC {}".format(band))
                                    imghst2hsc = gs.Convolve(
                                        realGalaxy.rotate(anglehst * gs.radians).shift(
                                            result.x[1], result.x[2]
                                        ), imgpsfgs).drawImage(
                                        nx=sizeCutout, ny=sizeCutout, scale=scalehsc)
                                    imgs = (img.array, "my naive"), (imghst2hsc.array, "GS RealGal")
                                    noisetoadd = np.sqrt(var)
                                    descpre = "HST {} - {}"
                                    for imgidx, (imgit, desc) in enumerate(imgs):
                                        ax2[1, 1+imgidx].imshow(np.log10(imgit))
                                        ax2[1, 1+imgidx].set_title(descpre.format(bandhst, desc))
                                        imgit += np.random.normal(scale=noisetoadd)
                                        ax2[0, 1 + imgidx].imshow(np.log10(imgit))
                                        ax2[0, 1 + imgidx].set_title((descpre + " + noise").format(
                                            bandhst, desc))

                                sigmainverse = 1.0/np.sqrt(var)
                                # The PSF is now HSTPSF*HSCPSF, and "truth" is the deconvolved HST image
                                psf = gs.InterpolatedImage(gs.Convolve(
                                    imgpsfgs, psfhst.rotate(anglehst*gs.degrees)).drawImage(
                                    nx=imgpsf.shape[1], ny=imgpsf.shape[0], scale=scalehscpsf)
                                )
                                psf.image /= np.sum(psf.image.array)
                            else:
                                mask = exposure.getMaskedImage().getMask()
                                mask = mask.array[ids[3]: ids[2], ids[1]: ids[0]]
                                var = var.array[ids[3]: ids[2], ids[1]: ids[0]]
                                img = copy.deepcopy(exposure.maskedImage.image)
                                if not useNoiseReplacer:
                                    footprint = measCat[row].getFootprint()
                                    img *= 0
                                    footprint.getSpans().copyImage(exposure.maskedImage.image, img)
                                psf = imgpsfgs
                                mask = img != 0
                                img = gs.Image(img[idshsc[3]: idshsc[2], idshsc[1]: idshsc[0]],
                                               scale=scalehsc)
                            sigmainverse = 1.0 / np.sqrt(var)

                    fits, models, psfmodels = fitgalaxy(
                        img=img, psf=psf, sigmainverse=sigmainverse, band=band, name="COSMOS #{}".format(id),
                        modellib=args.modellib, algo=args.algo, psfmodel=args.psfmodel, plot=args.plot)
                    for band in bands:
                        # Set all exposures to have no images - we can rebuild them if needed
                        for engine, modelinfos in models.items():
                            for modelname, modelinfo in modelinfos.items():
                                models[engine][modelname] = setexposure(modelinfo["model"], band)
                        for engine, modelinfo in psfmodels.items():
                            setexposure(modelinfo["model"], band)
                        # TODO: This might be a little too convoluted. Deconvolve.
                    for modelname, modelfitinfo in fits.items():
                        for engine, modelfits in modelfitinfo["fits"].items():
                            for fit in modelfits:
                                fit["fitinfo"]["log"] = None
                                if hasattr(fit["result"], "problem"):
                                    fit["result"] = None
                    data[id] = {"models": models, "fits": fits, "psfmodels": psfmodels}
            except Exception as e:
                print("Error fitting id={}:".format(id))
                print(e)
                trace = traceback.format_exc()
                print(trace)
                data[id] = e, trace

            nfit += 1
            if args.fileout is not None and (nfit % 10) == 0:
                with open(os.path.expanduser(args.fileout), 'wb') as f:
                    pickle.dump(data, f)
