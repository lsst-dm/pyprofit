from abc import ABCMeta, abstractmethod
import copy
import galsim as gs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pygmo as pg
import pyprofit as pyp
import scipy.stats as spstats
import scipy.optimize as spopt
import seaborn as sns
import time

#gaussian_sigma_to_fwhm = 2.35482004503094932701
#gaussian_fwhm_to_sigma = 0.424660900144009534340


# https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
# Can get better performance but this isn't critical as it's just being used to check if bands are identical
def allequal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

# TODO: Implement WCS
# The smart way for this to work would be to specify sky coordinates and angular sizes for objects. This
# way you could give a model some exposures with WCS and it would automagically convert to pixel coordinates.


# TODO: Make this a class
def getgsparams(engineopts):
    if engineopts is None:
        gsparams = gs.GSParams()
    else:
        gsparams = engineopts["gsparams"]
    return gsparams


class Exposure:
    """
        A class to hold an image, sigma map, bad pixel mask and reference to a PSF model/image
    """
    def __init__(self, band, image, maskinverse=None, sigmainverse=None, psf=None, calcinvmask=None, meta={}):
        if psf is not None and not isinstance(psf, PSF):
            raise TypeError("Exposure (band={}) PSF type={:s} not instanceof({:s})".format(
                band, type(psf), type(PSF)))
        self.band = band
        self.image = image
        self.maskinverse = maskinverse
        self.sigmainverse = sigmainverse
        self.psf = psf
        self.calcinvmask = calcinvmask
        self.meta = meta


class Data:
    """
        A class that holds exposures, mostly for convenience to pass to a model.
        Currently a dict keyed by band with items lists of exposures.
        Also enforces that all exposures are the same size right now, assuming that they're pixel matched.
        This will be relaxed once WCS is implemented.
        TODO: Should this use multiband exposures? Dunno
    """
    def __init__(self, exposures):
        self.exposures = {}
        self.nx = None
        for i, exposure in enumerate(exposures):
            if not isinstance(exposure, Exposure):
                raise TypeError(
                    "exposure[{:d}] (type={:s}) is not an instance of {:s}".format(
                        i, type(exposure), type(Exposure))
                )
            else:
                if self.nx is None:
                    self.nx = exposure.image.shape[0]
                    self.ny = exposure.image.shape[1]
                else:
                    if exposure.image.shape[0] != self.nx or exposure.image.shape[1] != self.ny:
                        raise ValueError(
                            "Mismatch in exposure[{:d}] (band={:s}] shape={} vs expected {}".format(
                                i, band, exposure.image.shape, (self.nx, self.ny)))
                band = exposure.band
                if band not in self.exposures:
                    self.exposures[band] = []
                self.exposures[band].append(exposure)


class PSF:
    """
        Either a model or an image.

        Has convenience functions to convert to/from galsim format.
    """

    def get(self, usemodel=None):
        if usemodel is None:
            usemodel = self.usemodel
        if usemodel:
            return self.model
        return self.getimage()

    def getimageshape(self):
        if self.image is None:
            return None
        if isinstance(self.image, gs.InterpolatedImage):
            return self.image.image.array.shape
        return self.image.shape

    # TODO: support rescaling of the PSF if it's a galsim interpolatedimage?
    def getimage(self, engine=None, size=None, engineopts=None):
        if engine is None:
            engine = self.engine
        if size is None and self.model is None:
            size = self.getimageshape()
        if self.image is None or self.getimageshape() != size:
            if self.model is None:
                raise RuntimeError("Can't get new PSF image without a model")
            exposure = Exposure(self.band, np.zeros(shape=size), None, None)
            data = Data(exposures=[exposure])
            model = Model(sources=[self.model], data=data)
            # TODO: Think about the consequences of making a new astrometry vs resetting the old one
            # It's necessary because the model needs to be centered and it might not be
            astro = model.sources[0].modelastrometric
            model.sources[0].modelastrometric = AstrometricModel([
                Parameter("cenx", value=size[0]/2.),
                Parameter("ceny", value=size[1]/2.),
            ])
            model.evaluate(keepimages=True, getlikelihood=False)
            self.image = data.exposures[self.band][0].meta["modelimage"]
            model.sources[0].modelastrometric = astro

        # TODO: There's more torturous logic here if we're to support changing engines on the fly
        if engine != self.engine:
            if engine == "galsim":
                gsparams = getgsparams(engineopts)
                self.image = gs.InterpolatedImage(gs.ImageD(self.image, scale=1), gsparams=gsparams)
            else:
                if self.engine == "galsim":
                    self.image = self.image.image.array
            self.engine = engine
        return self.image

    def __init__(self, band, image=None, model=None, engine=None, usemodel=False):
        self.band = band
        self.model = model
        self.image = image
        if model is None:
            if image is None:
                raise ValueError("PSF must be initialized with either a model or engine but both are none")
            if usemodel:
                raise ValueError("PSF usemodel==True but no model specified")
            if (engine == "galsim") and (isinstance(image, gs.InterpolatedImage) or
                                         isinstance(image, gs.Image)):
                self.engine = engine
            else:
                if not isinstance(image, np.ndarray):
                    raise ValueError("PSF image must be an ndarray or galsim.Image/galsim.InterpolatedImage if "
                                     "using galsim")
                self.engine = "libprofit"
                self.image = self.getimage(engine=engine)
        else:
            if image is not None:
                raise ValueError("PSF initialized with a model cannot be initialized with an image as well")
            if not isinstance(model, Source):
                raise ValueError("PSF model (type={:s}) not instanceof({:s})".format(
                    type(model), type(Source)))
            self.engine = engine


class Model:
    likefuncs = {
        "normal": "normal"
        , "gaussian": "normal"
        , "student-t": "t"
        , "t": "t"
    }

    ENGINES = ["galsim", "libprofit"]

    @classmethod
    def _checkengine(cls, engine):
        if engine not in Model.ENGINES:
            raise ValueError("Unknown Model rendering engine {:s}".format(engine))

    def evaluate(self, params=None, data=None, bands=None, engine=None, engineopts=None,
                 paramstransformed=True, getlikelihood=True, likelihoodlog=True, keeplikelihood=False,
                 keepimages=False, plot=False):
        """
            Get the likelihood and/or model images
        """
        if engine is None:
            engine = self.engine
        Model._checkengine(engine)
        if engineopts is None:
            engineopts = self.engineopts
        if data is None:
            data = self.data

        paramobjects = self.getparameters(free=True, fixed=False)
        if params is None:
            params = [param.value for param in paramobjects]
        else:
            paramslen = len(params)
            if not len(paramobjects) == paramslen:
                raise RuntimeError("Length of parameters[{:d}] != # of free params=[{:d}]".format(
                    len(paramobjects), paramslen
                ))
            for paramobj, paramval in zip(paramobjects, params):
                paramobj.setvalue(paramval, transformed=paramstransformed)

        if getlikelihood:
            likelihood = 1.
            if likelihoodlog:
                likelihood = 0.
        else:
            likelihood = None

        if bands is None:
            bands = data.exposures.keys()
        if plot:
            plotslen = 0
            for band in bands:
                plotslen += len(data.exposures[band])
            figure, axes = plt.subplots(nrows=plotslen + (plotslen == 3), ncols=5)
            if plotslen == 1:
                axes.shape = (1, 5)
            figurerow = 0
        else:
            figaxes = None
        chis = []
        chiclips = []
        chiimgs = []
        imgclips = []
        modelclips = []
        for band in bands:
            # TODO: Check band
            for exposure in data.exposures[band]:
                image = self.getexposuremodel(exposure, engine=engine, engineopts=engineopts)
                if keepimages:
                    exposure.meta["modelimage"] = np.array(image)
                if getlikelihood or plot:
                    if plot:
                        figaxes = (figure, axes[figurerow])
                        figurerow += 1
                    likelihoodexposure, chi, chiimg, chiclip, imgclip, modelclip = \
                        self.getexposurelikelihood(
                            exposure, image, log=likelihoodlog, figaxes=figaxes)
                    if keeplikelihood:
                        exposure.meta["likelihood"] = likelihoodexposure
                        exposure.meta["likelihoodlog"] = likelihoodlog
                    if likelihoodlog:
                        likelihood += likelihoodexposure
                    else:
                        likelihood *= likelihoodexposure
                    if plot and plotslen == 3:
                        chis.append(chi)
                        chiclips.append(chiclip)
                        chiimgs.append(chiimg)
                        imgclips.append(imgclip)
                        modelclips.append(modelclip)
            # Color images! whooo
        if plot:
            if plotslen == 3:
                self.plotexposurescolor(imgclips, modelclips, chis, chiimgs,
                                        chiclips, (figure, axes[figurerow]))
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
        return likelihood, params

    def plotexposurescolor(self, images, modelimages, chis, chiimgs, chiclips, figaxes):
        # TODO: verify lengths
        axes = figaxes[1]
        shapeimg = images[0].shape
        for i, imagesbytype in enumerate([images, modelimages]):
            rgb = np.zeros(shapeimg + (0,))
            for image in imagesbytype:
                rgb = np.append(rgb, image.reshape(shapeimg + (1,)), axis=2)
            axes[i].imshow(np.flip(rgb, axis=2), origin="bottom")
        rgb = np.zeros(shapeimg + (0,))
        for image in chiimgs:
            rgb = np.append(rgb, image.reshape(shapeimg + (1,)), axis=2)
        # The difference map
        axes[2].imshow(np.flip(rgb, axis=2), origin="bottom")
        # axes[2].set_title(r'$\chi^{2}_{\nu}$' + '={:.3f}'.format(chisqred))
        # The chi (data-model)/error map clipped at +/- 5 sigma
        rgb = np.zeros(shapeimg + (0,))
        for image in chiclips:
            rgb = np.append(rgb, image.reshape(shapeimg + (1,)), axis=2)
        axes[3].imshow(np.flip(rgb, axis=2), origin="bottom")
        # Residual histogram compared to a normal distribution
        chi = np.array([])
        for chiband in chis:
            chi = np.append(chi, chiband)
        nbins = np.max([100, np.int(np.round(np.sum(~np.isnan(chi))/300))])
        sns.distplot(chi[~np.isnan(chi)], bins=nbins, ax=axes[4],
                     hist_kws={"log": True, "histtype": "step"},
                     kde_kws={"kernel": "tri", "gridsize": nbins/2}).set(
            xlim=(-5, 5), ylim=(1e-4, 1)
        )
        # axes[4].hist(chi[~np.isnan(chi)], bins=100, log=True, density=True, histtype="step", fill=False)
        x = np.linspace(-5., 5., int(1e4) + 1, endpoint=True)
        axes[4].plot(x, spstats.norm.pdf(x))
        chisqred = (np.sum(chi ** 2)/len(chi))
        axes[4].set_title(r'$\chi^{2}_{\nu}$' + '={:.3f}'.format(chisqred))
        for i in range(1, 5):
            axes[i].set_yticklabels([])

    def getexposurelikelihood(self, exposure, modelimage, log=True, likefunc=None,
                              figaxes=None, maximg=None, minimg=None):
        if likefunc is None:
            likefunc = self.likefunc
        hasmask = exposure.maskinverse is not None
        if figaxes is not None:
            axes = figaxes[1]
            xlist = np.arange(0, modelimage.shape[1])
            ylist = np.arange(0, modelimage.shape[0])
            x, y = np.meshgrid(xlist, ylist)
            chi = (exposure.image - modelimage)
            if maximg is None:
                if hasmask:
                    maximg = np.max([np.max(exposure.image[exposure.maskinverse]),
                                     np.max(modelimage[exposure.maskinverse])])
                else:
                    maximg = np.max([np.max(exposure.image), np.max(modelimage)])
            if minimg is None:
                minimg = maximg/1e4
            # The original image and model image
            imgclips = []
            for i, img in enumerate([exposure.image, modelimage]):
                imgclip = (np.log10(np.clip(img, minimg, maximg)) - np.log10(minimg))/np.log10(maximg/minimg)
                imgclips.append(imgclip)
                axes[i].imshow(imgclip, cmap='gray', origin="bottom")
                axes[i].set_xticklabels([])
                if hasmask:
                    z = exposure.maskinverse
                    axes[i].contour(x, y, z)
            axes[0].set_title('Band={}'.format(exposure.band))
            axes[1].set_title('Model'.format(exposure.band))
            # The (logged) difference map
            chilog = np.log10(np.clip(np.abs(chi), minimg, np.inf)/minimg)*np.sign(chi)
            chilog /= np.log10(maximg/minimg)
            chilog = np.clip((chilog+1.)/2., 0, 1)
            axes[2].imshow(chilog, cmap='gray', origin='bottom')
            if hasmask:
                axes[2].contour(x, y, z)
            # The chi (data-model)/error map clipped at +/- 5 sigma
            chi *= exposure.sigmainverse
            chiclip = np.copy(chi)
            chiclip[chi < -5] = -5
            chiclip[chi > 5] = 5
            if hasmask:
                chi = chi[exposure.maskinverse]
                chisqred = (np.sum(chi**2) /
                            np.sum(exposure.maskinverse))
            else:
                chisqred = np.sum(chi**2) / np.prod(chi.shape)
            axes[3].imshow(chiclip, cmap='RdYlBu_r', origin="bottom")
            chiclip += 5.
            chiclip /= 10.
            if hasmask:
                axes[3].contour(x, y, z, colors="green")
            # Residual histogram compared to a normal distribution
            nbins = np.max([100, np.int(np.round(np.sum(~np.isnan(chi))/300))])
            sns.distplot(chi[~np.isnan(chi)], bins=nbins, ax=axes[4],
                         hist_kws={"log": True, "histtype": "step"},
                         kde_kws={"kernel": "tri", "gridsize": nbins/2}).set(
                xlim=(-5, 5), ylim=(1e-4, 1)
            )
            # axes[4].hist(chi[~np.isnan(chi)], bins=100, log=True, density=True, histtype="step", fill=False)
            x = np.linspace(-5., 5., int(1e4) + 1, endpoint=True)
            axes[4].plot(x, spstats.norm.pdf(x))
            axes[4].set_title(r'$\chi^{2}_{\nu}$' + '={:.3f}'.format(chisqred))
            axes[4].yaxis.tick_right()
            for i in range(1, 5):
                axes[i].set_yticklabels([])
        else:
            chilog = None
            chiclip = None
            imgclips = [None, None]
            if hasmask:
                chi = (exposure.image[exposure.maskinverse] - modelimage[exposure.maskinverse]) * \
                    exposure.sigmainverse[exposure.maskinverse]
            else:
                chi = (exposure.image - modelimage)*exposure.sigmainverse
        if likefunc == "t":
            variance = chi.var()
            dof = 2. * variance / (variance - 1.)
            dof = max(min(dof, float('inf')), 0)
            likelihood = np.sum(spstats.t.logpdf(chi, dof))
        elif likefunc == "normal":
            likelihood = np.sum(spstats.norm.logpdf(chi))
        else:
            raise ValueError("Unknown likelihood function {:s}".format(self.likefunc))

        if not log:
            likelihood = np.exp(likelihood)

        return likelihood, chi, chilog, chiclip, imgclips[0], imgclips[1]


    # TODO: Implement analytic Gaussian convolution
    def getexposuremodel(self, exposure, engine=None, engineopts=None):
        """
            Draw model image for one exposure with one PSF
        """
        if engine is None:
            engine = self.engine
        if engineopts is None:
            engineopts = self.engineopts
        Model._checkengine(engine)
        ny, nx = exposure.image.shape
        band = exposure.band
        profiles = self.getprofiles([band], engine=engine)

        haspsf = exposure.psf is not None
        allgaussian = haspsf and exposure.psf.model is not None and \
            all([comp.isgaussian() for comp in exposure.psf.model.modelphotometric.components]) and \
            all([comp.isgaussian() for comps in [src.modelphotometric.components for src in self.sources]
                 for comp in comps])
        if allgaussian:
            # TODO: actually implement this.
            pass

        if engine == "libprofit":
            profilespro = {}
            for profile in profiles:
                profile = profile[band]
                profiletype = profile["profile"]
                if profiletype not in profilespro:
                    profilespro[profiletype] = []
                del profile["profile"]
                if not profile["pointsource"]:
                    profile["convolve"] = haspsf
                    if not profile["resolved"]:
                        raise RuntimeError("libprofit can't handle non-point sources that aren't resolved"
                                           "(i.e. profiles with size==0)")
                del profile["pointsource"]
                del profile["resolved"]
                # TODO: Find a better way to do this
                for coord in ["x", "y"]:
                    nameold = "cen" + coord
                    profile[coord + "cen"] = profile[nameold]
                    del profile[nameold]
                profilespro[profiletype] += [profile]

            profit_model = {
                'width': nx,
                'height': ny,
                'magzero': 0.0,
                'profiles': profilespro,
            }
            if haspsf:
                shape = exposure.psf.getimageshape()
                if shape is None:
                    shape = [1 + np.int(x/2) for x in np.floor([nx, ny])]
                profit_model["psf"] = exposure.psf.getimage(engine, size=shape, engineopts=engineopts)

            if exposure.calcinvmask is not None:
                profit_model['calcmask'] = exposure.calcinvmask
            image = np.array(pyp.make_model(profit_model)[0])
        elif engine == "galsim":
            profilesgs = {
                True: None
                , False: None
            }
            cenimg = gs.PositionD(nx/2., ny/2.)
            for profile in profiles:
                profile = profile[band]
                if not profile["pointsource"] and not profile["resolved"]:
                    profile["pointsource"] = True
                    raise RuntimeError("Converting small profiles to point sources not implemented yet")
                    # TODO: Finish this
                else:
                    profilegs = profile["profile"].shear(
                        profile["shear"]).shift(
                        profile["offset"] - cenimg)
                    convolve = haspsf and not profile["pointsource"]
                if profilesgs[convolve] is None:
                    profilesgs[convolve] = profilegs
                else:
                    profilesgs[convolve] += profilegs

            gsparams = getgsparams(engineopts)
            if haspsf:
                psfgs = exposure.psf.model
                if psfgs is None:
                    profilespsf = exposure.psf.getimage(engine=engine)
                else:
                    psfgs = psfgs.getprofiles(bands=[band], engine=engine)
                    profilespsf = None
                    for profile in psfgs:
                        profile = profile[band]
                        profilegs = profile["profile"].shear(profile["shear"])
                        # TODO: Think about whether this would ever be necessary
                        #.shift(profile["offset"] - cenimg)
                        if profilespsf is None:
                            profilespsf = profilegs
                        else:
                            profilespsf += profilegs
                profilesgs[True] = gs.Convolve(profilesgs[True], profilespsf, gsparams=gsparams)
            else:
                if profilesgs[True] is not None:
                    raise RuntimeError("Model (band={}) has profiles to convolve but no PSF".format(
                        exposure.band))
            if self.engineopts is not None and "drawmethod" in self.engineopts:
                method = self.engineopts["drawmethod"]
            else:
                method = "fft"
            # Has a PSF, but it's not an analytic model, so it must be an image and therefore pixel
            # convolution is already included for extended objects, which must use no_pixel
            try:
                haspsfimage = haspsf and psfgs is None and profilesgs[True] is not None
                if haspsfimage:
                    imagegs = profilesgs[True].drawImage(method="no_pixel", nx=nx, ny=ny, scale=1)
                    if profilesgs[False] is not None:
                        imagegs += profilesgs[False].drawImage(method=method, nx=nx, ny=ny, scale=1)
                else:
                    if profilesgs[True] is not None:
                        if profilesgs[False] is not None:
                            profilesgs = profilesgs[True] + profilesgs[False]
                        else:
                            profilesgs = profilesgs[True]
                    else:
                        profilesgs = profilesgs[False]

            except RuntimeError as e:
                if method == "fft":
                    if haspsfimage:
                        imagegs = profilesgs[True].drawImage(method="real_space", nx=nx, ny=ny, scale=1) + \
                                  profilesgs[False].drawImage(method=method, nx=nx, ny=ny, scale=1)
                    else:
                        imagegs = profilesgs.drawImage(method="real_space", nx=nx, ny=ny, scale=1)
                else:
                    raise e
            except Exception as e:
                raise e
            # TODO: Determine why this is necessary. Should we keep an imageD per exposure?
            image = np.copy(imagegs.array)

        sumnotfinite = np.sum(~np.isfinite(image))
        if sumnotfinite > 0:
            raise RuntimeError("{}.getexposuremodel() got {:d} non-finite pixels out of {:d}".format(
                type(self), sumnotfinite, np.prod(image.shape)
            ))

        return image

    def getlimits(self, free=True, fixed=True, transformed=True):
        params = self.getparameters(free=free, fixed=fixed)
        return [param.getlimits(transformed=transformed) for param in params]

    def getprofiles(self, bands, engine=None, engineopts=None):
        '''

        :param bands: List of bands
        :param engine: Valid engine
        :param engineopts: Dict of engine options
        :return: List of profiles
        '''
        if engine is None:
            engine = self.engine
        if engineopts is None:
            engineopts = self.engineopts
        self._checkengine(engine)
        profiles = []
        for src in self.sources:
            profiles += src.getprofiles(engine=engine, bands=bands, engineopts=engineopts)
        return profiles

    def getparameters(self, free=True, fixed=True):
        params = []
        for src in self.sources:
            params += src.getparameters(free=free, fixed=fixed)
        return params

    def getparamnames(self, free=True, fixed=True):
        names = []
        for i, src in enumerate(self.sources):
            srcname = src.name
            if srcname == "":
                srcname = str(i)
            names += [".".join([srcname, paramname]) for paramname in \
                      src.getparamnames(free=free, fixed=fixed)]
        return names

    def getpriorvalue(self, free=True, fixed=True, log=True):
        return np.sum(np.array(
            [param.getprior(log=log) for param in self.getparameters(free=free, fixed=fixed)]
        ))

    # TODO: implement
    def getpriormodes(self, free=True, fixed=True):
        params = self.getparameters(free=free, fixed=fixed)
        return [param.getprior.mode for param in params]

    def getlikelihood(self, params=None, data=None, log=True, plot=False):
        likelihood = self.evaluate(params, data, likelihoodlog=log, plot=plot)[0]
        return likelihood

    def __init__(self, sources, data=None, likefunc="normal", engine=None, engineopts=None):
        if engine is None:
            engine = "libprofit"
        for i, source in enumerate(sources):
            if not isinstance(source, Source):
                raise TypeError(
                    "Model source[{:d}] (type={:s}) is not an instance of {:s}".format(
                        i, type(source), type(Source))
                )
        if data is not None and not isinstance(data, Data):
            raise TypeError(
                "Model data (type={:s}) is not an instance of {:s}".format(
                    str(type(data)), str(Data))
            )
        self.sources = sources
        self.data = data
        Model._checkengine(engine)
        self.engine = engine
        self.engineopts = engineopts
        self.likefunc = likefunc


class ModellerPygmoUDP:
    def fitness(self, x):
        return [-self.modeller.evaluate(x, returnlponly=True, timing=self.timing)]

    def get_bounds(self):
        return self.boundslower, self.boundsupper

    def gradient(self, x):
        return pg.estimate_gradient(self.fitness, x)

    def __init__(self, modeller, boundslower, boundsupper, timing=False):
        self.modeller = modeller
        self.boundslower = boundslower
        self.boundsupper = boundsupper
        self.timing = timing

    def __deepcopy__(self, memo):
        """
            This is some rather ugly code to make a deep copy of a modeller without copying the model's
            data, which should be shared between all copies of a model (and its modeller).
        """
        modelself = self.modeller.model
        model = Model(sources=copy.deepcopy(modelself.sources, memo=memo), data=modelself.data,
                      likefunc=copy.copy(modelself.likefunc), engine=copy.copy(modelself.engine),
                      engineopts=copy.copy(modelself.engineopts))
        modeller = Modeller(model=model, modellib=copy.copy(self.modeller.modellib),
                            modellibopts=copy.copy(self.modeller.modellibopts))
        modeller.fitinfo = copy.copy(self.modeller.fitinfo)
        copied = self.__class__(modeller=modeller, boundslower=copy.copy(self.boundslower),
                                boundsupper=copy.copy(self.boundsupper), timing=copy.copy(self.timing))
        memo[id(self)] = copied
        return copied


class Modeller:
    """
        A class that does the fitting for a Model.
        Some useful things it does is define optimization functions called by libraries and optionally
        store info that they don't track (mainly per-iteration info, including parameter values,
        running time, separate priors and likelihoods rather than just the objective, etc.).
    """
    # TODO: Implement
    def evaluate(self, paramsfree=None, timing=False, returnlponly=False, returnlog=True, plot=False):

        if timing:
            tinit = time.time()
        # TODO: Attempt to prevent/detect defeating this by modifying fixed/free params?
        prior = self.fitinfo["priorLogfixed"] + self.model.getpriorvalue(free=True, fixed=False)
        likelihood = self.model.getlikelihood(paramsfree, plot=plot)
        # return LL, LP, etc.
        if returnlponly:
            rv = likelihood + prior
            if not returnlog:
                rv = np.exp(rv)
        else:
            if not returnlog:
                likelihood = np.exp(likelihood)
                prior = np.exp(prior)
            rv = likelihood, prior
        if timing:
            tstep = time.time() - tinit
            rv += tstep
        loginfo = {
            "params": paramsfree
            , "likelihood": likelihood
            , "prior": prior
        }
        if timing:
            loginfo["time"] = tstep
            loginfo["tinit"] = tinit
        if "log" in self.fitinfo:
            self.fitinfo["log"].append(loginfo)
        if ("printsteps" in self.fitinfo and
                self.fitinfo["printsteps"] is not None):
            stepnum = len(self.fitinfo["log"])
            if stepnum % self.fitinfo["printsteps"] == 0:
                print(stepnum, rv, loginfo)
        return rv

    def fit(self, paramsinit=None, printfinal=True, timing=None, maxwalltime=np.Inf, printsteps=None):
        # TODO: Calculate log prior of non-fitted params
        self.fitinfo["priorLogfixed"] = self.model.getpriorvalue(free=False, fixed=True, log=True)
        self.fitinfo["log"] = []
        self.fitinfo["printsteps"] = printsteps

        if paramsinit is None:
            paramsinit = [param.getvalue(transformed=True) for param in self.model.getparameters(fixed=False)]
            # TODO: Why did I think I would want to do this?
            #paramsinit = self.model.getpriormodes(free=True, fixed=False)

        # TODO: Finish implementing this
        #paramnames = self.model.getparamnames(fixed=False)
        paramnames = [param.name for param in self.model.getparameters(fixed=False)]
        print("Param names:\n", paramnames)
        print("Initial parameters:\n", paramsinit)
        print("Evaluating initial parameters:\n", self.evaluate(paramsinit))

        timerun = 0.0

        limits = self.model.getlimits(fixed=False, transformed=True)
        algo = self.modellibopts["algo"]

        if self.modellib == "scipy":
            def neg_like_model(params, modeller):
                return -modeller.evaluate(params, timing=timing, returnlponly=True)

            tinit = time.time()
            result = spopt.minimize(neg_like_model, paramsinit, method=algo,
                                    bounds=np.array(limits),
                                    options={'disp': True}, args=(self, ))
            timerun += time.time() - tinit
            paramsbest = result.x

        elif self.modellib == "pygmo":
            algocmaes = algo == "cmaes"
            algonlopt = not algocmaes
            if algocmaes:
                uda = pg.cmaes()
            elif algonlopt:
                uda = pg.nlopt(algo)
                uda.ftol_abs = 1e-3
                if np.isfinite(maxwalltime) and maxwalltime > 0:
                    uda.maxtime = maxwalltime
                nloptopts = ["stopval", "ftol_rel", "ftol_abs", "xtol_rel", "xtol_abs", "maxeval"]
                for nloptopt in nloptopts:
                    if nloptopt in self.modellibopts and self.modellibopts[nloptopt] is not None:
                        setattr(uda, nloptopt, self.modellibopts[nloptopt])

            algo = pg.algorithm(uda)
            #        algo.extract(pg.nlopt).ftol_rel = 1e-6
            if algonlopt:
                algo.extract(pg.nlopt).ftol_abs = 1e-3

            if "verbosity" in self.modellibopts and self.modellibopts["verbosity"] is not None:
                algo.set_verbosity(self.modellibopts["verbosity"])
            limitslower = np.zeros(len(limits))
            limitsupper = np.zeros(len(limits))
            for i, limit in enumerate(limits):
                limitslower[i] = limit[0]
                limitsupper[i] = limit[1]

            udp = ModellerPygmoUDP(modeller=self, boundslower=limitslower, boundsupper=limitsupper,
                                   timing=timing)
            problem = pg.problem(udp)
            pop = pg.population(prob=problem, size=0)
            if algocmaes:
                npop = 5
                npushed = 0
                while npushed < npop:
                    try:
                        #pop.push_back(init + np.random.normal(np.zeros(np.sum(data.tofit)),
                        #                                      data.sigmas[data.tofit]))
                        npushed += 1
                    except:
                        pass
            else:
                pop.push_back(paramsinit)
            tinit = time.time()
            result = algo.evolve(pop)
            timerun += time.time() - tinit
            paramsbest = result.champion_x
        else:
            raise RuntimeError("Unknown optimization library " + self.modellib)

        if printfinal:
            print("Elapsed time: {:.1f}".format(timerun))
            print("Final likelihood: {}".format(self.evaluate(paramsbest)))
            print("Parameter names:        " + ",".join(["{:10s}".format(i) for i in paramnames]))
            print("Transformed parameters: " + ",".join(["{:.4e}".format(i) for i in paramsbest]))
            # TODO: Finish
            #print("Parameters (linear): " + ",".join(["{:.4e}".format(i) for i in paramstransformed]))

        result = {
            "fitinfo": copy.copy(self.fitinfo),
            "paramsbest": paramsbest,
            "result": result,
            "time": timerun,
        }
        return result

    # TODO: Should constraints be implemented?
    def __init__(self, model, modellib, modellibopts, constraints=None):
        self.model = model
        self.modellib = modellib
        self.modellibopts = modellibopts
        self.constraints = constraints

        # Scratch space, I guess...
        self.fitinfo = {
            "priorLogfixed": np.log(1.0)
        }

class Source:
    """
        A model of a source, like a galaxy or star, or even the PSF (TBD).
    """
    ENGINES = ["galsim", "libprofit"]

    @classmethod
    def _checkengine(cls, engine):
        if engine not in Model.ENGINES:
            raise ValueError("Unknown {:s} rendering engine {:s}".format(type(cls), engine))

    def getparameters(self, free=None, fixed=None, time=None):
        astrometry = self.modelastrometric.getposition(time)
        return self.modelastrometric.getparameters(free, fixed, time) + \
            self.modelphotometric.getparameters(free, fixed, astrometry=astrometry)

    # TODO: Finish this
    def getparamnames(self, free=None, fixed=None):
        return self.modelastrometric.getparamnames(free, fixed) + \
            self.modelphotometric.getparameters(free, fixed)

    def getprofiles(self, engine, bands, time=None, engineopts=None):
        self._checkengine(engine)
        cenx, ceny = self.modelastrometric.getposition(time=time)
        return self.modelphotometric.getprofiles(engine=engine, bands=bands, cenx=cenx, ceny=ceny, time=time,
                                                 engineopts=engineopts)

    def __init__(self, modelastrometry, modelphotometry, name=""):
        self.name = name
        self.modelphotometric = modelphotometry
        self.modelastrometric = modelastrometry


class PhotometricModel:
    def getparameters(self, free=True, fixed=True, astrometry=None):
        params = [flux for flux in self.fluxes.values() if
                  (flux.fixed and fixed) or (not flux.fixed and free)]
        for comp in self.components:
            params += comp.getparameters(free=free, fixed=fixed)
        return params

    def getprofiles(self, engine, bands, cenx, ceny, time=None, engineopts=None):
        '''

        :param engine:
        :param bands:
        :param cenx:
        :param ceny:
        :param time:
        :param engineopts:
        :return: List of dicts by band
        '''
        # TODO: Check if this should skip entirely instead of adding a None for non-included bands
        bandfluxes = {band: self.fluxes[band].getvalue(transformed=False) if
                      band in self.fluxes else None for band in bands}
        profiles = []
        for comp in self.components:
            profiles += comp.getprofiles(bandfluxes, engine, cenx, ceny, engineopts=engineopts)
        return profiles

    def __init__(self, components, fluxes=[]):
        for i, comp in enumerate(components):
            if not isinstance(comp, Component):
                raise TypeError("PhotometricModel component[{:s}](type={:s}) "
                                "is not an instance of {:s}".format(
                    i, type(comp), type(Component)))
        for i, flux in enumerate(fluxes):
            if not isinstance(flux, FluxParameter):
                raise TypeError("PhotometricModel flux[{:d}](type={:s}) is not an instance of {:s}".format(
                    i, type(flux), type(FluxParameter)))
        bandscomps = [[flux.band for flux in comp.fluxes] for comp in components]
        # TODO: Check if component has a redundant mag or no specified flux ratio
        if not allequal(bandscomps):
            raise ValueError(
                "Bands of component fluxes in PhotometricModel components not all equal: {}".format(
                    bandscomps))
        bandsfluxes = [flux.band for flux in fluxes]
        if any([band not in bandscomps[0] for band in bandsfluxes]):
            raise ValueError("Bands of fluxes in PhotometricModel fluxes not all in fluxes of the "
                             "components: {} not all in {}".format(bandsfluxes, bandscomps[0]))
        self.components = components
        self.fluxes = {flux.band: flux for flux in fluxes}


# TODO: Implement and use, with optional WCS attached?
class Position:
    def __init__(self, x, y):
        for key, value in {"x":x, "y":y}:
            if not isinstance(value, Parameter):
                raise TypeError("Position[{:s}](type={:s}) is not an instance of {:s}".format(
                    key, type(param), type(Parameter)))
        self.x = x
        self.y = y


class AstrometricModel:
    """
        The astrometric model for this source.
        TODO: Implement moving models, or at least think about how to do it
    """

    def getparameters(self, free=True, fixed=True, time=None):
        return [value for value in self.params.values() if \
                (value.fixed and fixed) or (not value.fixed and free)]

    def getposition(self, time=None):
        return self.params["cenx"].getvalue(transformed=False), self.params["ceny"].getvalue(
            transformed=False)

    def __init__(self, params):
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                raise TypeError("Mag[{:d}](type={:s}) is not an instance of {:s}".format(
                    i, type(param), type(Parameter)))
            # TODO: Check if component has a redundant mag or no specified flux ratio
        self.params = {param.name: param for param in params}


# TODO: Store position and/or astrometry
class Component(object, metaclass=ABCMeta):
    """
        A component of a source, which can be extended or not. This abstract class only stores fluxes.
        TODO: Implement shape model or at least alternative angle/axis ratio implementations (w/boxiness)
        It could be isophotal twisting, a 3D shape, etc.
    """
    optional = ["cenx", "ceny"]

    @abstractmethod
    def getprofiles(self, bandfluxes, engine, cenx, ceny, engineopts=None):
        """
            bandfluxes is a dict of bands with item flux in a linear scale or None if components independent
            Return is dict keyed by band with lists of engine-dependent profiles.
            galsim are GSObjects
            libprofit are dicts with a "profile" key
        """
        pass

    @abstractmethod
    def getparameters(self, free=True, fixed=True):
        pass

    @abstractmethod
    def isgaussian(self):
        pass

    def __init__(self, fluxes, name=""):
        for i, param in enumerate(fluxes):
            if not isinstance(param, Parameter):
                raise TypeError(
                    "Component param[{:d}] (type={:s}) is not an instance of {:s}".format(
                        i, type(param), type(Parameter))
                )
        self.fluxes = fluxes
        self.name = name


class EllipticalProfile(Component):
    """
        Class for any profile with a (generalized) ellipse shape.
        TODO: implement boxiness for libprofit; not sure if galsim does generalized ellipses?
    """
    profilesavailable = ["moffat", "sersic"]
    mandatoryshape = ["ang", "axrat"]
    # TODO: Consider adopting gs's flexible methods of specifying re, fwhm, etc.
    mandatory = {
        "moffat": mandatoryshape + ["con", "fwhm"]
        , "sersic": mandatoryshape + ["nser", "re"]
    }

    ENGINES = ["galsim", "libprofit"]

    @classmethod
    def _checkengine(cls, engine):
        if engine not in Model.ENGINES:
            raise ValueError("Unknown {:s} rendering engine {:s}".format(type(cls), engine))

    # TODO: Should the parameters be stored as a dict? This method is the only reason why it's useful now
    def isgaussian(self):
        return (self.profile == "sersic" and self.parameters["nser"].getvalue() == 0.5) \
            or (self.profile == "moffat" and np.isinf(self.parameters["con"].getvalue()))

    def getparameters(self, free=True, fixed=True):
        return [value for value in self.fluxes if \
                (value.fixed and fixed) or (not value.fixed and free)] + \
            [value for value in self.parameters.values() if \
                (value.fixed and fixed) or (not value.fixed and free)]

    def getprofiles(self, bandfluxes, engine, cenx, ceny, engineopts=None):
        '''

        :param bandfluxes: Dict of fluxes by band
        :param engine: Rendering engine
        :param cenx: X center in image coordinates
        :param ceny: Y center in image coordinates
        :param engineopts: Dict of engine options
        :return: Dict by band with list of profiles
        '''
        self._checkengine(engine)
        isgaussian = self.isgaussian()

        fluxesbands = {flux.band: flux for flux in self.fluxes}
        for band in bandfluxes.keys():
            if band not in fluxesbands:
                raise ValueError(
                    "Asked for EllipticalProfile (profile={:s}, name={:s}) model for band={:s} not in "
                    "bands with fluxes {}".format(self.profile, self.name, band, fluxesbands))

        profiles = {}
        for band in bandfluxes.keys():
            flux = fluxesbands[band].getvalue(transformed=False)
            if fluxesbands[band].isfluxratio:
                fluxratio = copy.copy(flux)
                if not 0 <= fluxratio <= 1:
                    raise ValueError("flux ratio not 0 <= {} <= 1".format(fluxratio))
                flux *= bandfluxes[band]
                # bandfluxes[band] -= flux
                # TODO: Is subtracting as above best? Should be more accurate, but mightn't guarantee flux>=0
                bandfluxes[band] *= (1.0-fluxratio)
            profile = {param.name: param.getvalue(transformed=False) for param in
                       self.parameters.values()}
            if not 0 < profile["axrat"] <= 1:
                if profile["axrat"] > 1:
                    profile["axrat"] = 1
                elif profile["axrat"] <= 0:
                    profile["axrat"] = 1e-3
                else:
                    raise ValueError("axrat {} ! >0 and <=1".format(profile["axrat"]))

            cens = {"cenx": cenx, "ceny": ceny}
            # Does this profile have a non-zero size?
            # TODO: Review cutoff - it can't be zero for galsim or it will request huge FFTs
            resolved = not (
                (self.profile == "sersic" and profile["re"] < 0*(engine == "galsim")) or
                (self.profile == "moffat" and profile["fwhm"] < 0*(engine == "galsim"))
            )
            for key, value in cens.items():
                if key in profile:
                    profile[key] += value
                else:
                    profile[key] = copy.copy(value)
            if resolved:
                if engine == "galsim":
                    axrat = profile["axrat"]
                    axratsqrt = np.sqrt(axrat)
                    gsparams = getgsparams(engineopts)
                    if isgaussian:
                        if self.profile == "sersic":
                            fwhm = 2.*profile["re"]
                        else:
                            fwhm = profile["fwhm"]
                        profilegs = gs.Gaussian(
                            flux=flux, fwhm=fwhm*axratsqrt,
                            gsparams=gsparams
                        )
                    elif self.profile == "sersic":
                        if profile["nser"] < 0.3 or profile["nser"] > 6.2:
                            print("Warning: Sersic n {:.3f} not >= 0.3 and <= 6.2; "
                                  "GalSim could fail.".format(profile["nser"]))
                        profilegs = gs.Sersic(
                            flux=flux, n=profile["nser"],
                            half_light_radius=profile["re"]*axratsqrt,
                            gsparams=gsparams
                        )
                    elif self.profile == "moffat":
                        profilegs = gs.Moffat(
                            flux=flux, beta=profile["con"],
                            fwhm=profile["fwhm"]*axratsqrt,
                            gsparams=gsparams
                        )
                    profile = {
                        "profile": profilegs,
                        "shear": gs.Shear(g=(1.-axrat)/(1.+axrat), beta=(profile["ang"] + 90.)*gs.degrees),
                        "offset": gs.PositionD(profile["cenx"], profile["ceny"]),
                    }
                elif engine == "libprofit":
                    profile["mag"] = -2.5 * np.log10(flux)
                    # TODO: Review this. It might not be a great idea because Sersic != Moffat integration
                    # libprofit should be able to handle Moffats with infinite con
                    if self.profile != "sersic" and self.isgaussian():
                        profile["profile"] = "sersic"
                        profile["nser"] = 0.5
                        if self.profile == "moffat":
                            profile["re"] = profile["fwhm"]/2.0
                            del profile["fwhm"]
                            del profile["con"]
                        else:
                            raise RuntimeError("No implentation for turning profile {:s} into gaussian".format(
                                profile))
                    else:
                        profile["profile"] = self.profile
                else:
                    raise ValueError("Unimplemented rendering engine {:s}".format(engine))
            else:
                profile["flux"] = flux
            # This profile is part of a point source *model*
            profile["pointsource"] = False
            profile["resolved"] = resolved
            profiles[band] = profile
        return [profiles]

    @classmethod
    def _checkparameters(cls, parameters, profile):
        mandatory = {param: False for param in EllipticalProfile.mandatory[profile]}
        paramnamesneeded = mandatory.keys()
        paramnames = [param.name for param in parameters]
        errors = []
        # Not as efficient as early exit if true but oh well
        if len(paramnames) > len(set(paramnames)):
            errors.append("Parameters array not unique")
        # Check if these parameters are known (in mandatory)
        for param in parameters:
            if isinstance(param, FluxParameter):
                errors.append("Param {:s} is {:s}, not {:s}".format(param.name, type(FluxParameter),
                                                                    type(Parameter)))
            if param.name in paramnamesneeded:
                mandatory[param.name] = True
            elif param.name not in Component.optional:
                errors.append("Unknown param {:s}".format(param.name))

        for paramname, found in mandatory.items():
            if not found:
                errors.append("Missing mandatory param {:s}".format(paramname))
        if errors:
            errorstr = "Errors validating params of component (profile={:s}):\n" + \
                       "\n".join(errors) + "\nPassed params:" + str(parameters)
            raise ValueError(errorstr)

    def __init__(self, fluxes, name="", profile="sersic", parameters=None):
        if profile not in EllipticalProfile.profilesavailable:
            raise ValueError("Profile type={:s} not in available: ".format(profile) + str(
                EllipticalProfile.profilesavailable))
        self._checkparameters(parameters, profile)
        self.profile = profile
        Component.__init__(self, fluxes, name)
        self.parameters = {param.name: param for param in parameters}


def normalize(array):
    array /= np.sum(array)
    return array


class MultiGaussianApproximationProfile(Component):
    # TODO: Figure out a better way to split functionality between this and EllipticalProfile
    # Maybe have a generic EllipticalProfile and SingleEllipticalProfile?
    """
        Class for multi-Gaussian profiles with (generalized) ellipse shape(s).
    """
    profilesavailable = ["sersic", "lux", "luv"]
    mandatoryshape = ["ang", "axrat"]
    ENGINES = ["galsim", "libprofit"]
    mandatoryshape = ["ang", "axrat"]
    mandatory = {
        "sersic": mandatoryshape + ["nser", "re"],
        "lux": mandatoryshape + ["re"],
        "luv": mandatoryshape + ["re"],
    }

    weights = {
        1: {
            4: (normalize(np.array([0.09733, 1.12804, 4.99846, 5.63632])),
                np.array([0.12068, 0.32730, 0.68542, 1.28089])),
            6: (normalize(np.array([0.00735, 0.09480, 0.63570, 2.60073, 5.42849, 3.16451])),
                np.array([0.05072, 0.13756, 0.28781, 0.53194, 0.91208, 1.50156])),
            8: (normalize(np.array([0.00077, 0.01017, 0.07313, 0.13580, 0.25096, 1.39727,
                                    3.56054, 4.74340])),
                np.array([0.02393, 0.06490, 0.37188, 0.42942, 0.69672, 1.08879,
                                    1.78732, 1.67294]))
        },
        4: {
            6: (normalize(np.array([0.01308, 0.12425, 0.63551, 2.22560, 5.63989, 9.81523])),
                np.array([0.00263, 0.01202, 0.04031, 0.12128, 0.36229, 1.23604])),
            8: (normalize(np.array([0.00262, 0.02500, 0.13413, 0.51326, 1.52005, 3.56205,
                                    6.44845, 8.10105])),
                np.array([0.00113, 0.00475, 0.01462, 0.03930, 0.09926, 0.24699,
                                    0.63883, 1.92560])),
            10: (normalize(np.array([0.00139, 0.00941, 0.04441, 0.16162, 0.48121, 1.20357,
                                     2.54182, 4.46441, 6.22821, 6.15393])),
                 np.array([0.00087, 0.00296, 0.00792, 0.01902, 0.04289, 0.09351,
                                     0.20168, 0.44126, 1.01833,2.74555])),
        }
    }

    @classmethod
    def _checkengine(cls, engine):
        if engine not in Model.ENGINES:
            raise ValueError("Unknown {:s} rendering engine {:s}".format(type(cls), engine))

    def isgaussian(self):
        return True

    def getparameters(self, free=True, fixed=True):
        return [value for value in self.fluxes if \
                (value.fixed and fixed) or (not value.fixed and free)] + \
            [value for value in self.parameters.values() if \
                (value.fixed and fixed) or (not value.fixed and free)]

    def getprofiles(self, bandfluxes, engine, cenx, ceny, engineopts=None):
        self._checkengine(engine)

        fluxesbands = {flux.band: flux for flux in self.fluxes}
        for band in bandfluxes.keys():
            if band not in fluxesbands:
                raise ValueError(
                    "Asked for EllipticalProfile (profile={:s}, name={:s}) model for band={:s} not in "
                    "bands with fluxes {}".format(self.profile, self.name, band, fluxesbands))

        profiles = [{} for _ in range(self.order)]
        for band in bandfluxes.keys():
            flux = fluxesbands[band].getvalue(transformed=False)
            if fluxesbands[band].isfluxratio:
                fluxratio = copy.copy(flux)
                if not 0 <= fluxratio <= 1:
                    raise ValueError("flux ratio not 0 <= {} <= 1".format(fluxratio))
                flux *= bandfluxes[band]
                # bandfluxes[band] -= flux
                # TODO: Is subtracting as above best? Should be more accurate, but mightn't guarantee flux>=0
                bandfluxes[band] *= (1.0-fluxratio)
            profile = {param.name: param.getvalue(transformed=False) for param in
                       self.parameters.values()}
            if not 0 < profile["axrat"] <= 1:
                if profile["axrat"] > 1:
                    profile["axrat"] = 1
                elif profile["axrat"] <= 0:
                    profile["axrat"] = 1e-3
                else:
                    raise ValueError("axrat {} ! >0 and <=1".format(profile["axrat"]))

            cens = {"cenx": cenx, "ceny": ceny}
            for key, value in cens.items():
                if key in profile:
                    profile[key] += value
                else:
                    profile[key] = copy.copy(value)
            if engine == "galsim":
                axrat = profile["axrat"]
                axratsqrt = np.sqrt(axrat)
                gsparams = getgsparams(engineopts)
            elif engine == "libprofit":
                profile["profile"] = "sersic"
                profile["nser"] = 0.5
            else:
                raise ValueError("Unimplemented rendering engine {:s}".format(engine))
            profile["pointsource"] = False
            profile["resolved"] = True

            keyweight = profile["nser"] if self.profile == "sersic" else self.profile
            weightsprofile = MultiGaussianApproximationProfile.weights[keyweight][self.order]
            for subcomp, (weight, sigma) in enumerate(zip(weightsprofile[0], weightsprofile[1])):
                weightprofile = copy.copy(profile)
                if engine == "galsim":
                    profilegs = gs.Gaussian(
                        flux=weight*flux, fwhm=2.0*profile["re"]*axratsqrt*sigma,
                        gsparams=gsparams
                    )
                    weightprofile.update({
                        "profile": profilegs,
                        "shear": gs.Shear(g=(1.-axrat)/(1.+axrat), beta=(profile["ang"] + 90.)*gs.degrees),
                        "offset": gs.PositionD(profile["cenx"], profile["ceny"]),
                    })
                elif engine == "libprofit":
                    weightprofile["mag"] = -2.5 * np.log10(weight * flux)
                profiles[subcomp][band] = weightprofile

        return profiles

    @classmethod
    def _checkparameters(cls, parameters, profile, order):
        mandatory = {param: False for param in EllipticalProfile.mandatory[profile]}
        paramnamesneeded = mandatory.keys()
        paramnames = [param.name for param in parameters]
        errors = []
        if len(paramnames) > len(set(paramnames)):
            errors.append("Parameters array not unique")
        # Check if these parameters are known (in mandatory)

        if profile != "sersic" and order not in MultiGaussianApproximationProfile.weights[profile]:
            raise ValueError("{} profile={} order={} not in supported {}".format(
                self.__class__.__name__, profile, order,
                MultiGaussianApproximationProfile.weights[profile].keys()))

        for param in parameters:
            if isinstance(param, FluxParameter):
                errors.append("Param {:s} is {:s}, not {:s}".format(param.name, type(FluxParameter),
                                                                    type(Parameter)))
            if param.name in paramnamesneeded:
                mandatory[param.name] = True
                if param.name == "nser":
                    nser = param.getvalue(transformed=False)
                    if nser not in MultiGaussianApproximationProfile.weights:
                        raise ValueError("{} profile={} {}={} not in supported {}".format(
                            self.__class__.__name__, profile, param.name, nser,
                            MultiGaussianApproximationProfile.weights.keys()))
                    if order not in MultiGaussianApproximationProfile.weights[nser]:
                        raise ValueError("{} profile={} ({}={}) order={} not in supported {}".format(
                            self.__class__.__name__, profile, param.name, nser, order,
                            MultiGaussianApproximationProfile.weights[nser].keys()))
            elif param.name not in Component.optional:
                errors.append("Unknown param {:s}".format(param.name))

        for paramname, found in mandatory.items():
            if not found:
                errors.append("Missing mandatory param {:s}".format(paramname))
        if errors:
            errorstr = "Errors validating params of component (profile={:s}):\n" + \
                       "\n".join(errors) + "\nPassed params:" + str(parameters)
            raise ValueError(errorstr)

    def __init__(self, fluxes, name="", profile="exp", parameters=None, order=6):
        if profile not in MultiGaussianApproximationProfile.profilesavailable:
            raise ValueError("Profile type={:s} not in available: ".format(profile) + str(
                MultiGaussianApproximationProfile.profilesavailable))
        self._checkparameters(parameters, profile, order)
        self.profile = profile
        self.order = order
        Component.__init__(self, fluxes, name)
        self.parameters = {param.name: param for param in parameters}


class PointSourceProfile(Component):

    ENGINES = ["galsim", "libprofit"]

    @classmethod
    def _checkengine(cls, engine):
        if engine not in Model.ENGINES:
            raise ValueError("Unknown {:s} rendering engine {:s}".format(type(cls), engine))

    @classmethod
    def isgaussian(cls):
        return False

    def getparameters(self, free=True, fixed=True):
        return [value for value in self.fluxes if \
                (value.fixed and fixed) or (not value.fixed and free)]

    # TODO: default PSF could be unit image?
    def getprofiles(self, bandfluxes, engine, cenx, ceny, psf=None):
        """

        :param bandfluxes:
        :param engine:
        :param cenx:
        :param ceny:
        :param psf: A PSF (required, despite the default).
        :return:
        """
        self._checkengine(engine)
        if not isinstance(psf, PSF):
            raise TypeError("")

        fluxesbands = {flux.band: flux for flux in self.fluxes}
        for band in bandfluxes.keys():
            if band not in fluxesbands:
                raise ValueError(
                    "Called PointSourceProfile (name={:s}) getprofiles() for band={:s} not in "
                    "bands with fluxes {}", self.name, band, fluxesbands)

        # TODO: Think of the best way to do this
        # TODO: Ensure that this is getting copies - it isn't right now
        profiles = psf.model.getprofiles(engine=engine, bands=bandfluxes.keys())
        for band in bandfluxes.keys():
            flux = fluxesbands[band].getvalue(transformed=False)
            for profile in profiles[band]:
                if engine == "galsim":
                    profile["profile"].flux *= flux
                elif engine == "libprofit":
                    profile["mag"] -= 2.5 * np.log10(flux)
                profile["pointsource"] = True
            else:
                raise ValueError("Unimplemented PointSourceProfile rendering engine {:s}".format(engine))
        return profiles

    def __init__(self, fluxes, name=""):
        Component.__init__(fluxes=fluxes, name=name)


class Transform:
    @classmethod
    def null(cls, value):
        return value

    def __init__(self, transform=None, reverse=None):
        if transform is None or reverse is None:
            if transform is not reverse:
                raise ValueError(
                    "One of transform (type={:s}) and reverse (type={:s}) is {:s} but "
                    "both or neither must be".format(type(transform, type(reverse), type(None)))
                 )
            else:
                transform = self.null
                reverse = self.null
        self.transform = transform
        self.reverse = reverse
        # TODO: Verify if forward(reverse(x)) == reverse(forward(x)) +/- error for x in ???

class Limits:
    """
        Limits for a Parameter.
    """
    def within(self, value):
        return self.lower <= value <= self.upper

    def __init__(self, lower=-np.inf, upper=np.inf, lowerinclusive=True, upperinclusive=True,
                 transformed=True):
        isnanlower = np.isnan(lower)
        isnanupper = np.isnan(upper)
        if isnanlower or isnanupper:
            raise ValueError("Limits lower,upper={},{} finite check={},{}".format(
                lower, upper, isnanlower, isnanupper))
        if not upper >= lower:
            raise ValueError("Limits upper={} !>= lower{}".format(lower, upper))
        # TODO: Should pass in the transform and check if lower
        if not lowerinclusive:
            lower = np.nextafter(lower, lower+1.)
        if not upperinclusive:
            upper = np.nextafter(upper, upper-1.)
        self.lower = lower
        self.upper = upper
        self.transformed = transformed


# TODO: This class needs loads of sanity checks and testing
class Parameter:
    """
        A parameter with all the info about itself that you would need when fitting.
    """
    def getvalue(self, transformed=False):
        value = copy.copy(self.value)
        if transformed and not self.transformed:
            value = self.transform.transform(value)
        elif not transformed and self.transformed:
            value = self.transform.reverse(value)
        return value

    def getprior(self, log=True):
        if self.prior is None:
            prior = 1.0
            if log:
                prior = 0.
        else:
            prior = self.prior.getvalue(param=self, log=log)
        return prior

    # TODO: Decide what to do about inclusiveness
    def getlimits(self, transformed=False):
        lower = self.limits.lower
        upper = self.limits.upper
        if transformed and not self.limits.transformed:
            lower = self.transform.transform(lower)
            upper = self.transform.transform(upper)
        elif not transformed and self.limits.transformed:
            lower = self.transform.reverse(lower)
            upper = self.transform.reverse(upper)
        return lower, upper

    def setvalue(self, value, transformed=False):
        if not transformed:
            if value < self.limits.lower:
                value = self.limits.lower
        if transformed and not self.transformed:
            value = self.transform.reverse(value)
        elif not transformed and self.transformed:
            value = self.transform.transform(value)
        self.value = value

    def __init__(self, name, value, unit="", limits=None, transform=Transform(), transformed=True, prior=None,
                 fixed=False):
        if prior is not None and not isinstance(prior, Prior):
            raise TypeError("prior (type={:s}) is not an instance of {:s}".format(type(prior), type(Prior)))
        if limits is None:
            limits = Limits(transformed=transformed)
        if limits.transformed != transformed:
            raise ValueError("limits.transformed={} != Param[{:s}].transformed={}".format(
                limits.transformed, name, transformed
            ))
        self.fixed = fixed
        self.name = name
        self.value = value
        self.unit = unit
        self.limits = limits
        self.transform = transform
        self.transformed = transformed
        self.prior = prior


class FluxParameter(Parameter):
    """
        A flux, magnitude or flux ratio, all of which one could conceivably fit.
        TODO: name seems a bit redundant, but I don't want to commit to storing the band as a string now
    """
    def __init__(self, band, name, value, unit, limits, transform=None, transformed=True, prior=None,
                 fixed=None, isfluxratio=None):
        if isfluxratio is None:
            isfluxratio = False
        Parameter.__init__(self, name=name, value=value, unit=unit, limits=limits, transform=transform,
                           transformed=transformed, prior=prior, fixed=fixed)
        self.band = band
        self.isfluxratio = isfluxratio


class Prior:
    """
        A prior probability distribution function.
        Not an ecclesiastical superior usually of lower rank than an abbot.

        TODO: I'm not sure how to enforce proper normalization without implementing specific subclasses
        e.g. Even in a flat prior, if the limits change the normalization does too.
    """
    def getvalue(self, param, log):
        if not isinstance(param, Parameter):
            raise TypeError(
                "param(type={:s}) is not an instance of {:s}".format(type(param), type(Parameter)))

        if self.transformed != self.limits.transformed:
            raise ValueError("Prior must have same transformed flag as its Limits")

        paramval = param.getvalue(transformed=self.transformed)
        if self.limits.within(paramval):
            prior = self.func(paramval)
        else:
            prior = 0.0
        if log and not self.log:
            return np.log(prior)
        elif not log and self.log:
            return np.exp(prior)
        return prior

    def __init__(self, func, log, transformed, mode, limits):
        if not isinstance(limits, Limits):
            "Prior limits(type={:s}) is not an instance of {:s}".format(type(limits), type(Limits))
        # TODO: how to type check this?
        self.func = func
        self.log = log
        self.transformed = transformed
        self.mode = mode
        self.limits = limits
