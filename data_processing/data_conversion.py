import numpy as np
from sklearn import svm
from scipy.io import loadmat
import os
from multiprocessing import Pool
from os import listdir
import pandas as pd
import gzip
import json
import time
import cv2
import matplotlib.pyplot as plt

data_directory = "K:/dataset"
converted_directory = "K:/dataset_simplified/radar_images"

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

# We are provided these files by the resources section on the KRadar Github
# Arr Doppler represent the Doppler Bins the RADAR outputs
arrDoppler = loadmat("./arr_doppler.mat")["arr_doppler"]
arrDopplerLookup = np.array(arrDoppler[0])

# Arr Info, similar to Arr Doppler represents binning information, however
# It contains range, elevation, and azimuth information. It is believed
# that they acquired doppler by a secondary means and that is why it is split
arrInfo = loadmat("./info_arr.mat")
radar_range  = np.array(arrInfo["arrRange"][0])
arrElevationLookup = np.array(arrInfo["arrElevation"][0])
radar_azimuth = np.array(arrInfo["arrAzimuth"][0]*np.pi/180)


'''
By looking at info_arr.mat, we were able to find the dimensions of our tesseract arrays which represent the binned-data responses.

I.e the ranges are binned in X divisions, we can access what would be in Doppler Bin 1, Range Bin 1, Elevation Bin 1, and Azimuth Bin 1 by accesing arrDREA[0][0][0][0]
It is up to us to decode those bins back into real numbers for our model. The real question is whether or not we want to flatten our array.

I.e convert the 4-D array into a 2D array with rows being arbitrary and the other information just being encoded on the secondary axis
'''
# Using the obtained Lookup information we will now begin to convert rows in tesseracts.
# Tesseracts are the 4D radar's raw information
# Array goes 4 deep: Doppler -> Range -> Elevation -> Azimuth -> Intensity (Actual value held by the array)
# Doppler x Range x Elevation x Azimuth
def process_tesseract(tesseract):
    print("Beginning Conversion of: " + tesseract[:-4])
    start = time.time()
    
    input_path = tesseract
    output_path = os.path.join(converted_directory, os.path.basename(tesseract)[:-4] + ".png")
    arr_DREA = np.array(loadmat(input_path)['arrDREA'])
    radar_tensor = np.transpose(arr_DREA, (0, 1, 3, 2))
    radar_bev = np.mean(np.mean(radar_tensor, axis=0), axis=2)

    arr_0, arr_1 = np.meshgrid(radar_azimuth, radar_range)
    height, width = np.shape(radar_bev)


    max_val = np.max(10*np.log10(radar_bev))
    min_val = np.min(10*np.log10(radar_bev))

    with open(output_path[:-4] + ".txt", 'w') as min_max:
        min_max.writelines(str(max_val) + "\n")
        min_max.writelines(str(min_val))

    plt.clf()
    plt.cla()
    plt.pcolormesh(arr_0, arr_1, 10*np.log10(radar_bev), cmap='gray')
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)

    temp_img = cv2.imread(output_path)
    temp_row, temp_col, _ = temp_img.shape
    if not (temp_row == height and temp_col == width):
        temp_img_new = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(output_path, temp_img_new)

    plt.close()
    plt.pcolormesh(arr_0, arr_1, 10*np.log10(radar_bev), cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)

    end = time.time()
    print("Conversion and saving of {} completed in {:.2f} seconds.".format(tesseract[:-4],end - start))

def obtain_tesseract_paths(dataDirectory):
    return [os.path.join(dataDirectory, "radar_tesseract", file) for file in listdir(os.path.join(dataDirectory, "radar_tesseract"))]


    
if __name__ == "__main__":
    tesseractNames = obtain_tesseract_paths(data_directory)
    with Pool(processes=10) as pool:
        pool.map(process_tesseract, tesseractNames)
    
    
# This converter is really only ran once and should not need to be used in a "live" environment.