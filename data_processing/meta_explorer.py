import os

meta_directory = 'K:/datasequence_metas'
label_directory = 'info_label'

def get_labels(directory):
    label_directory_path = meta_directory + "/" + directory + "/" + label_directory
    list_labels = os.listdir(label_directory_path)
    print(list_labels)



if __name__ == '__main__':
    list_metas = os.listdir(meta_directory)
    for meta in list_metas:
        get_labels(meta)