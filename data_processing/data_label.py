'''
This script takes the handcropped dataset imagery and quantifies them
into the following values: Size X, Size Y, Total Size, Average Intensity, Maximum Intensity,
and Minimum Intensity. The only data output really used from this script at this
time is the Total Size and Average Intensity.


Author: Kyler Farrar (kaf386@msstate.edu)
'''
import os
import numpy as np
from scipy import interpolate
from PIL import Image
import pandas as pd

# Label Directory
labeled_data_directory = '/mnt/KRadar/dataset_simplified/radar_labeled/'
radar_data_directory = "/mnt/KRadar/dataset_simplified/radar_images/"
output_data_directory = '/mnt/KRadar/dataset_simplified/radar_quantified/'


def get_child_dirs(parent_dir):
    # Obtains a list of sub-directories while ignoring files
    return [name for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]

def get_child_files(dir):
    # Thumbs.db is sometimes generated depending on the operating system being used. As such, we ignore that file if it exists
    return [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name)) and name != "Thumbs.db"]

def closeness_to_max_white(RGBA_value):
    # Gets a "mean" percentage of the RGBA value of the pixel to 255, 255, 255, 255.
    # 255, 255, 255, 255 is white which in our radar images would represent max intensity
    elementwise_percentage_diff = (RGBA_value / 255) # Scale of 0 to 1 (since our max val is 255, this maintains a scale of 0 to 1)
    overall_diff = np.mean(elementwise_percentage_diff) # Mean of all differences as our returned percentage
    return overall_diff
    
if __name__ == "__main__":
    # We use pandas to help manage creating our CSV and for visualization purposes
    pandas_datadict = {"Dataset": [], "Frame Number": [], "Label": [], "Max Intensity": [], "Min Intensity": [], "Avg Intensity": [], "X Size": [], "Y Size": [], "Total Size": []}
    
    # This gets us into the label folders: I.e Bike, Motorcycle, Pedestrian
    for directory in get_child_dirs(labeled_data_directory):
        # This identifies the files within each label directory and loops over them
        
        # Print out the label (directory) and the number of files within it
        print("Processing " + directory + ": " + str(len(get_child_files(labeled_data_directory + directory))))
        for labeled_file in get_child_files(labeled_data_directory + directory):
            
            # Reconstruct file path using directory information
            label_file = labeled_data_directory + directory + "/" + labeled_file
            
            # File Name Format: DatasetNumber_FrameNumber_Label_ID.png
            # Split on underscores for parsing
            file_components = labeled_file.split("_")
            dataset_number = file_components[0]
            frame_number = file_components[1]
            label_type = file_components[2]
            
            
            # Build radar_images path to tesseract data reconsturction information: min intensity (0, 0, 0, 255 RGBA), max intensity (255, 255, 255, 255 RGBA)
            reconstruction_path = radar_data_directory + dataset_number + "/tesseract_" + frame_number + ".txt"
            
            # We acquire this min and max in the data_conversion stage and save it in a .txt file
            # with the same name as the considered .png file. 
            reconstruction_min = 0.
            reconstruction_max = 0.
            with open(reconstruction_path, 'r') as reconstruction_file:
                min_and_max = reconstruction_file.read().splitlines()
                reconstruction_min = np.double(min_and_max[1])
                reconstruction_max = np.double(min_and_max[0])
            
            # This gives a continous inteperation from our pixel "percent white" value to our intensity reconstruction values
            conversion_factor = interpolate.interp1d([0, 1], [reconstruction_min, reconstruction_max])
            
            # Begin recording tracked parameters
            max_intensity = reconstruction_max
            min_intensity = reconstruction_min
            avg_intensity = 0.
            size_x = 0
            size_y = 0
            total_size = 0.
            
            # Utilizes PIL to open the image
            with Image.open(label_file) as label_data:
                # Converts it to a numpy array for data operations
                img_np = np.asarray(label_data)
                size_x = img_np.shape[0]
                size_y = img_np.shape[1]
                total_size = size_x * size_y
                
                sum_intensities = 0.
                for x in range(size_x):
                    for y in range(size_y):
                        sum_intensities = sum_intensities + conversion_factor(closeness_to_max_white(img_np[x][y]))
                
                avg_intensity = sum_intensities / total_size
            
            # Store as a row in the dataset dictionary. Dataset and Frame Number are used as reference/lookup information.
            pandas_datadict["Dataset"].append(dataset_number)
            pandas_datadict["Frame Number"].append(frame_number)
            pandas_datadict["Label"].append(label_type)
            pandas_datadict["Max Intensity"].append(max_intensity)
            pandas_datadict["Min Intensity"].append(min_intensity)
            pandas_datadict["Avg Intensity"].append(avg_intensity)
            pandas_datadict["X Size"].append(size_x)
            pandas_datadict["Y Size"].append(size_y)
            pandas_datadict["Total Size"].append(total_size)
            
    df = pd.DataFrame(data=pandas_datadict).to_csv(output_data_directory + 'kradar_processed.csv', ',')
    
            
    