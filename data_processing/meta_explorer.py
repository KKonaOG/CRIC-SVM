'''
This script is somewhat of a one-off utility that reads through the meta information
provided in the K-Radar Google Drive. The meta folder's there contain the labelling information
for the relevant data sequence. As such, we are able to examine the data distribution and label
variety without requiring the entire K-Radar dataset.


Author: Kyler Farrar (kaf386@msstate.edu)
'''

import os

'''
The directory which holds all the datasequence metas downloaded from the K-Radar Google Drive is the meta_directory. 
It is assumed that the datasequence metas are unzipped in the following structure:

meta_directory -
    - 1_meta
        - info_calib
        - info_label
        - time_info
    - 2_meta
        - info_calib
        - info_label
        - time_info
    - 3_meta
        - info_calib
        - info_label
        - time_info
    - 4_meta
        - info_calib
        - info_label
        - time_info
    - 5_meta
        - info_calib
        - info_label
        - time_info
    .
    .
    .
    - X_meta
        - info_calib
        - info_label
        - time_info

The only important folder used by the meta_explorer script is the info_label folder (which can be renamed as long as label_directory is reassigned) within each meta. All other folders within a meta are irrelevant.
'''
meta_directory = 'K:/datasequence_metas'

# Label Directory is simply the name of the folder containing the labels inside each "meta folder"
label_directory = 'info_label'

def get_labels(directory):
    # Reconstructs label base path
    label_directory_path = meta_directory + "/" + directory + "/" + label_directory

    # Obtains a list of the label files in the base path for that meta's labels
    list_labels = os.listdir(label_directory_path)

    # Dictionary of Labels and their Occurences
    labels = {}

    # Loop over each file in the list of labels for a meta
    for label_file in list_labels:
        with open(label_directory_path + "/" + label_file, 'r') as label:
            label.readline() # Skip First Line as it is header informatio

            # Look at each line in the file and select out the classification (should always be element 3 when you split by commas)
            # If the classification exists in the dictionary, increase the number of occurences associated with it
            # If the classification does not exist in the dictionary, create it and set its occurences to 1
            for line in label:
                key = str(line.split(',')[3].strip())
                if (not labels.get(key)):
                    labels[key] = 1
                else:
                    labels[key] = labels[key] + 1

    # Print out the breakdown for the file for analysis
    print("Meta: {}".format(directory))
    print(labels)
    print("======================")


if __name__ == '__main__':
    # Obtain list of directories within meta_directory
    list_metas = os.listdir(meta_directory)

    # Obtain labels within each meta folder
    for meta in list_metas:
        get_labels(meta)