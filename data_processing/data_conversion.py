'''
This script is a one-off utility that takes in the raw 4D-RADAR tensors
and converts them to db-Scale image representations of the tensor alongside
reconstruction information in a specified directory. This is the primary
automated stage of pre-processing. Following this stage, images are hand-cropped
in order to a fully labelled dataset for the SVM.

Author: Kyler Farrar (kaf386@msstate.edu)
'''

# Fundamental Requirements of the Conversion Process
import os
import numpy as np
from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt

# These are only required for multi-processing operations
from multiprocessing import Pool
import time

'''
data_directory is an individual datasequence from the K-Radar dataset.
It should at the minimum contain the radar_tesseract and info_labels folders from
the datasequence
'''
data_directory = "K:/dataset"

'''
converted_directory is the output location for the processed (converted) radar tensors
in the form of .png files. These .png files are Range (Height) and Angle (Width) correlated.
Importantly, elevation and doppler information have the intensity information averaged and act
as a color normalization value in grayscale when saved. In order to re-obtain data mappings from
this color mapping, minimum and maximum values of this axis are saved alongside the converted frames
for use in reconstruction.
'''
converted_directory = "K:/dataset_simplified/radar_images"

'''
info_arr.mat is obtained from the resources section of the K-Radar GitHub and
contains a bin number to physical world vaue mapping for Range, Azimuth, and Elevation.
Importantly, arr_doppler.mat is additionally available in the resources section,
however is not required in this pre-processing stage.
'''
radar_info = loadmat("./info_arr.mat") # Elevation is not needed in this case
radar_range  = np.array(radar_info["arrRange"][0])
radar_azimuth = np.array(radar_info["arrAzimuth"][0]*np.pi/180) # Azimuth is stored in degrees so we convert it back to radians

'''
    arrDREA's are the "code-name" assigned to RADAR tesseracts stored inside .mat files within the radar_tesseracts folder
    of a datasequence. They are stored in order of Doppler -> Range -> Azimith -> Elevation. Thus arrDREA[0][0][0][0] would
    return the intensity value for Doppler Bin 0, Range Bin 0, Azimuth Bin 0, and Elevation Bin 0.

    The actual process used in process_tesseract is a near exact duplicate of the process utilized inside K-Radar's GitHub
    to convert the RADAR tesseracts to .bev files. An important difference is that they utilize it for data visualization
    with bounding boxes, while, we use it for the purposes of obtaining data inputs for our models. 
    (See: https://github.com/kaist-avelab/K-Radar/blob/c2eb194b5ba2e08177ddc59fb97510e3ebde2188/utils/util_dataset.py#L34)
'''
def process_tesseract(tesseract):

    # Purely for logging purposes
    print("Beginning Conversion of: " + tesseract[:-4])
    start = time.time()
    
    # Tesseract is the full file-path of the desired .mat file
    input_path = tesseract

    # Set our output file to be in the converted directory folder with the same name as the original but instead of .mat, utilize .png
    output_path = os.path.join(converted_directory, os.path.basename(tesseract)[:-4] + ".png")

    # Load the .mat file (this can take a minute and requires a good bit of RAM)
    arr_DREA = np.array(loadmat(input_path)['arrDREA'])

    # Reorganize the tensor to be in: Doppler -> Range -> Elevation -> Azimuth Order
    radar_tensor = np.transpose(arr_DREA, (0, 1, 3, 2))

    # Obtain Intensity means along the Doppler Axis and Elevation Axis
    # This results in a Range by Angle Grid with Intensity means as the value
    radar_bev = np.mean(np.mean(radar_tensor, axis=0), axis=2)

    arr_0, arr_1 = np.meshgrid(radar_azimuth, radar_range)
    height, width = np.shape(radar_bev)

    # Obtain Max and Min Values for the purposes of reconstruction file
    max_val = np.max(10*np.log10(radar_bev))
    min_val = np.min(10*np.log10(radar_bev))

    # Save Reconstruction File
    with open(output_path[:-4] + ".txt", 'w') as min_max:
        min_max.writelines(str(max_val) + "\n")
        min_max.writelines(str(min_val))


    # Draw the image file utilize grayscale normalization for the intensity values
    # Importantly, the intensity values are scaled into dB values to avoid unnecessarily large
    # values. This was nearly directly copied and pasted from K-Radar's function.

    # TODO: Investigate usage of dpi=300 and its impacts.
    plt.clf()
    plt.cla()
    plt.pcolormesh(arr_0, arr_1, 10*np.log10(radar_bev), cmap='gray')
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)

    # Verify succesful image writing using OpenCV, importantly, CUBIC interpolation is
    # utilized to map the data back to a more dense form if required.
    temp_img = cv2.imread(output_path)
    temp_row, temp_col, _ = temp_img.shape
    if not (temp_row == height and temp_col == width):
        temp_img_new = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(output_path, temp_img_new)

    # Finally, replot the data a last time without any axis information, etc.
    # This is the last step of the conversion process and gives us a finalized
    # converted image.
    plt.close()
    plt.pcolormesh(arr_0, arr_1, 10*np.log10(radar_bev), cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)

    # Purely for logging purposes
    end = time.time()
    print("Conversion and saving of {} completed in {:.2f} seconds.".format(tesseract[:-4],end - start))

def obtain_tesseract_paths(dataDirectory):
    # Obtains the full directory path information for all .mat files in the radar_tesseract
    # for the given data_directory
    return [os.path.join(dataDirectory, "radar_tesseract", file) for file in os.listdir(os.path.join(dataDirectory, "radar_tesseract"))]


    
if __name__ == "__main__":
    # Obtains the full directory path information for all .mat files in the radar_tesseract
    # for the given data_directory
    tesseractNames = obtain_tesseract_paths(data_directory)

    # Utilizes 10 cores in a Pool to work on the tesseracts obtained by the obtain_tesseract_paths function
    with Pool(processes=10) as pool:
        pool.map(process_tesseract, tesseractNames)
    
    
# This converter is really only ran once and should not need to be used in a "live" environment.