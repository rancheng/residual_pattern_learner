import numpy as np


SOLVER_SVD = 1
SOLVER_ORTHOGONALIZE_SYSTEM = 2
SOLVER_ORTHOGONALIZE_POINTMARG = 4
SOLVER_ORTHOGONALIZE_FULL = 8
SOLVER_SVD_CUT7 = 16
SOLVER_REMOVE_POSEPRIOR = 32
SOLVER_USE_GN = 64
SOLVER_FIX_LAMBDA = 128
SOLVER_ORTHOGONALIZE_X = 256
SOLVER_MOMENTUM = 512
SOLVER_STEPMOMENTUM = 1024
SOLVER_ORTHOGONALIZE_X_LATER = 2048

PYR_LEVELS = 6
pyrLevelsUsed = PYR_LEVELS
patternNum = 8


staticPattern = np.array([
		[[0,0], 	  [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],	# .
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100]],

		[[0,-1],	  [-1,0],	   [0,0],	    [1,0],	     [0,1], 	  [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],	# +
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100]],

		[[-1,-1],	  [1,1],	   [0,0],	    [-1,1],	     [1,-1], 	  [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],	# x
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100]],

		[[-1,-1],	  [-1,0],	   [-1,1],		[-1,0],		 [0,0],		  [0,1],	   [1,-1],		[1,0],		 [1,1],       [-100,-100],	# full-tight
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100]],

		[[0,-2],	  [-1,-1],	   [1,-1],		[-2,0],		 [0,0],		  [2,0],	   [-1,1],		[1,1],		 [0,2],       [-100,-100],	# full-spread-9
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100]],

		[[0,-2],	  [-1,-1],	   [1,-1],		[-2,0],		 [0,0],		  [2,0],	   [-1,1],		[1,1],		 [0,2],       [-2,-2],   # full-spread-13
		 [-2,2],      [2,-2],      [2,2],       [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100]],

		[[-2,-2],     [-2,-1], [-2,-0], [-2,1], [-2,2], [-1,-2], [-1,-1], [-1,-0], [-1,1], [-1,2], 										# full-25
		 [-0,-2],     [-0,-1], [-0,-0], [-0,1], [-0,2], [+1,-2], [+1,-1], [+1,-0], [+1,1], [+1,2],
		 [+2,-2], 	  [+2,-1], [+2,-0], [+2,1], [+2,2], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100]],

		[[0,-2],	  [-1,-1],	   [1,-1],		[-2,0],		 [0,0],		  [2,0],	   [-1,1],		[1,1],		 [0,2],       [-2,-2],   # full-spread-21
		 [-2,2],      [2,-2],      [2,2],       [-3,-1],     [-3,1],      [3,-1], 	   [3,1],       [1,-3],      [-1,-3],     [1,3],
		 [-1,3],      [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100]],

		[[0,-2],	  [-1,-1],	   [1,-1],		[-2,0],		 [0,0],		  [2,0],	   [-1,1],		[0,2],		 [-100,-100], [-100,-100],	# 8 for SSE efficiency
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100],
		 [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100], [-100,-100]],

		[[-4,-4],     [-4,-2], [-4,-0], [-4,2], [-4,4], [-2,-4], [-2,-2], [-2,-0], [-2,2], [-2,4], 										# full-45-SPREAD
		 [-0,-4],     [-0,-2], [-0,-0], [-0,2], [-0,4], [+2,-4], [+2,-2], [+2,-0], [+2,2], [+2,4],
		 [+4,-4], 	  [+4,-2], [+4,-0], [+4,2], [+4,4], [-200,-200], [-200,-200], [-200,-200], [-200,-200], [-200,-200],
		 [-200,-200], [-200,-200], [-200,-200], [-200,-200], [-200,-200], [-200,-200], [-200,-200], [-200,-200], [-200,-200], [-200,-200]],
])

patternP = staticPattern[8]
# Parameters controlling when KF's are taken
setting_keyframesPerSecond = 0   # if !=0, takes a fixed number of KF per second.
setting_realTimeMaxKF = False   # if True, takes as many KF's as possible (will break the system if the camera stays stationary)
setting_maxShiftWeightT = 0.04 * (640+480)
setting_maxShiftWeightR = 0.0 * (640+480)
setting_maxShiftWeightRT = 0.02 * (640+480)
setting_kfGlobalWeight = 1   # general weight on threshold, the larger the more KF's are taken (e.g., 2 = double the amount of KF's).
setting_maxAffineWeight= 2

# initial hessian values to fix unobservable dimensions / priors on affine lighting parameters.

setting_idepthFixPrior = 50*50
setting_idepthFixPriorMargFac = 600*600
setting_initialRotPrior = 1e11
setting_initialTransPrior = 1e10
setting_initialAffBPrior = 1e14
setting_initialAffAPrior = 1e14
setting_initialCalibHessian = 5e9

# some modes for solving the resulting linear system (e.g. orthogonalize wrt. unobservable dimensions)

setting_solverMode = SOLVER_FIX_LAMBDA | SOLVER_ORTHOGONALIZE_X_LATER
setting_solverModeDelta = 0.00001
setting_forceAceptStep = True

# some thresholds on when to activate / marginalize points
setting_minIdepthH_act = 100
setting_minIdepthH_marg = 50


setting_desiredImmatureDensity = 1500 # immature points per frame
setting_desiredPointDensity = 2000 # aimed total points in the active window.
setting_minPointsRemaining = 0.05  # marg a frame if less than X% points remain.
setting_maxLogAffFacInWindow = 0.7 # marg a frame if factor between intensities to current frame is larger than 1/X or X.


setting_minFrames = 5 # min frames in window.
setting_maxFrames = 7 # max frames in window.
setting_minFrameAge = 1
setting_maxOptIterations=6 # max GN iterations.
setting_minOptIterations=1 # min GN iterations.
setting_thOptIterations=1.2 # factor on break threshold for GN iteration (larger = break earlier)

# Outlier Threshold on photometric energy
setting_outlierTH = 12*12					# higher -> less strict
setting_outlierTHSumComponent = 50*50 		# higher -> less strong gradient-based reweighting.

setting_pattern = 8						# point pattern used. DISABLED.
setting_margWeightFac = 0.5*0.5          # factor on hessian when marginalizing, to account for inaccurate linearization points.

setting_reTrackThreshold = 1.5          # (larger = re-track more often)

setting_minGoodActiveResForMarg=3
setting_minGoodResForMarg=4

setting_photometricCalibration = 2
setting_useExposure = True
setting_affineOptModeA = 1e12 #-1: fix. >=0: optimize (with prior, if > 0).
setting_affineOptModeB = 1e8 #-1: fix. >=0: optimize (with prior, if > 0).

setting_gammaWeightsPixelSelect = 0 # 1 = use original intensity for pixel selection 0 = use gamma-corrected intensity.




setting_huberTH = 9 # Huber Threshold


# parameters controlling adaptive energy threshold computation.
setting_frameEnergyTHConstWeight = 0.5
setting_frameEnergyTHN = 0.7
setting_frameEnergyTHFacMedian = 1.5
setting_overallEnergyTHWeight = 1
setting_coarseCutoffTH = 20


# parameters controlling pixel selection
setting_minGradHistCut = 0.5
setting_minGradHistAdd = 7
setting_gradDownweightPerLevel = 0.75
setting_selectDirectionDistribution = True

# settings controling initial immature point tracking
setting_maxPixSearch = 0.027 # max length of the ep. line segment searched during immature point tracking. relative to image resolution.
setting_minTraceQuality = 3
setting_minTraceTestRadius = 2
setting_GNItsOnPointActivation = 3
setting_trace_stepsize = 1.0				# stepsize for initial discrete search.
setting_trace_GNIterations = 3				# max # GN iterations
setting_trace_GNThreshold = 0.1				# GN stop after this stepsize.
setting_trace_extraSlackOnTH = 1.2			# for energy-based outlier check, be slightly more relaxed by this factor.
setting_trace_slackInterval = 1.5			# if pixel-interval is smaller than this, leave it be.
setting_trace_minImprovementFactor = 2		# if pixel-interval is smaller than this, leave it be.

minUseGrad_pixsel = 10

# for benchmarking different undistortion settings
benchmarkSetting_fxfyfac = 0
benchmarkSetting_width = 0
benchmarkSetting_height = 0
benchmark_varNoise = 0
benchmark_varBlurNoise = 0
benchmark_initializerSlackFactor = 1
benchmark_noiseGridsize = 3


freeDebugParam1 = 1
freeDebugParam2 = 1
freeDebugParam3 = 1
freeDebugParam4 = 1
freeDebugParam5 = 1

disableReconfigure=False
debugSaveImages = False
multiThreading = True
disableAllDisplay = False
setting_onlyLogKFPoses = True
setting_logStuff = True



goStepByStep = False


setting_render_displayCoarseTrackingFull=True
setting_render_renderWindowFrames=True
setting_render_plotTrackingFull = False
setting_render_display3D = True
setting_render_displayResidual = True
setting_render_displayVideo = True
setting_render_displayDepth = True

setting_fullResetRequested = False

setting_debugout_runquiet = False

sparsityFactor = 5  # not actually a setting, only some legacy stuff for coarse initializer.

staticPatternNum = [
		1,
		5,
		5,
		9,
		9,
		13,
		25,
		21,
		8,
		25
]

staticPatternPadding = [
		1,
		1,
		1,
		1,
		2,
		2,
		2,
		3,
		2,
		4
]
