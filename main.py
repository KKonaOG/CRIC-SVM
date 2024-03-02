import numpy as np
from sklearn import svm

'''
=============== STAGE 1: PRE-PROCESSING ===============
In order to work on the dataset, we first must generate our features.
What does this mean exactly? Well, we are provided a 4D RADAR Tensor
and a set of labels classiying the data within those tensors. We should
for each frame do two things:

One: Sort the frame into a Train, Test, Verification grouping for later purposes (60/30/10, 50/40/10, etc.)
Two:

Using the bounding boxes provided by the labelled dataset, find the relevant slices of data within the
frame. Until we have the dataset and know what data is available for the radar frame, it can be assumed we will be able
to extract the following: Azimuth, Elevation, Range, and Doppler. These values may or may not need to be averaged for the
boxed object. CRIC stands for Clustered Radar Intensity Classification which in this case can likely be boiled down to the
avergae of the doppler return for the region of interest. So at the very least we will need that feature. 

Stage 1 Pre-Processing should in the end result in three datasets (Train, Test, Verification) containing the feature data
for the frames in those datasets, this will be fed into Stage 2 which handles the SVMs being fit to the feature sets.
=======================================================
'''