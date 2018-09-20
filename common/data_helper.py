import glob
import pickle
from shutil import copy
from tqdm import tqdm


class DataHelper:
    """
    helpers to transform and move data around add more as needed.
    """

    def copy_specific_training_data_to_new_folder(self, source_folder_path, destination_folder_path,
                                                  track_name, track_variation):
        """
        Copies filtered raw data from source_folder_path to destination_folder_path.

        Keyword arguments:

        source_folder_path -- where the dat will be read from

        destination_folder_path -- where the filtered data will be saved 

        track_name -- filter by track name

        track_variation -- filter by track variation (e.g short)
        """
        listing = glob.glob(source_folder_path + '/*.png')

        for filename in tqdm(listing):
            filename = filename.replace('\\', '/')
            filename = filename.replace('-image.png', '')

            with open(filename + '-data.pkl', 'rb') as data:
                project_cars_state = pickle.load(data)
                #controller_state = pickle.load(data)

            # only do Watkins Glen International track data
            current_track = str(project_cars_state.mTrackLocation).replace(
                "'", "").replace("b", "")
            current_track_variation = str(
                project_cars_state.mTrackVariation).replace("'", "").replace("b", "")

            # if not on the correct track goto next track. *variation: #Short Circuit or #Grand Prix
            if(current_track != track_name and current_track_variation != track_variation):
                continue

            copy(filename + '-data.pkl', destination_folder_path)
            copy(filename + '-image.png', destination_folder_path)


#copy_specific_training_data_to_new_folder('F:/Project_Cars_Data/Raw',
#'F:/Project_Cars_Data/Watkins Glen International - Short Circuit',
#  'Watkins Glen International', 'Short Circuit')

# b'Watkins Glen International'
# b'Short Circuit'

# b'Watkins Glen International'
# b'Grand Prix'