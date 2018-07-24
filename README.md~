# local-linear-segmentation
This repository contains the main scripts for local linear segmentation and subsequent analysis of the resulting model space


To run the segmentation algorithm, the following steps must be followed:

1. - Install the following packages:

- scipy, numpy, matplotlib
- cython, scikit-learn, h5py


2. - run the 'setup.py' file which compiles the 'LLSA_calculations.pyx' cython script

python setup.py build_ext --inplace


In order to the able to run the .ipynb tutorials, you'll also need jupyter and seaborn.


(a) - 'SegmentingHO.ipynb' and (b) - 'SegmentingWormBehavior.ipynb' are two complementary tutorials on how to apply the adaptive locally-linear segmentation. In (a), a toy time series is segmented and hierarchical clustering is applied to obtain the original model parameters. In (b), a sample C. elegans "eigenworm" time series is analysed, in which the worm is subject to a heat shock to the head that triggers an escape response. The sampled time series start at the initiation of the escape response, which is broadly composed of a reversal, a turn, and forward movement in a different direction, away from the stimulus. 



-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------


'TestSegmentationScript.py' is a full python script that segments the toy time series. It takes various parameters as arguments
	- n: size of the null distribution
	- min_w: minimum window size
	- s: step fraction in the definition of the minimum window size
	- per: percentile of the null likelihood distribution that defines the threshold significance level (2x(100-per))

e.g.,

python TestSegmentationScript.py -n 1000 -min_w 6 -s 0.1 -per 97.5

'TestSegmentationScript.py' applies the segmentation algorithm to the time series from 'SampleTseries.h5'. The details of the simulation can be found in the 'MetaData' folder of 'SampleTseries.h5' and for more intuition follow the 'SegmentingHO.ipynb' tutorial.

'LLSA.py' contains the main skeleton of the segmentation algorithm

'LLSA_calculations.pyx' is a cython script with the main functions used in the segmentation (such as "get_theta" which fits a linear model, or "R_null" which draws the null likelihood distribution)

The resulting model space can be analysed by likelihood hierarchical clustering using 'Distance_calculations.py', which computes the likelihood distance matrix, performs hierarchical clustering, and returns the corresponding models
