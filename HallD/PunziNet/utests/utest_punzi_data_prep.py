import punzinet_toolkit.data_parser as parsers
import punzinet_toolkit.data_prep as preps
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class UTestPunziDataPrep(unittest.TestCase):

    # Initialize:
    #*****************************************
    def __init__(self,*args, **kwargs):
        super(UTestPunziDataPrep,self).__init__(*args, **kwargs)

        # Get an into:
        print(" ")
        print("*********************************")
        print("*                               *")
        print("*   Unit Test: Punzi Data Prep  *")
        print("*                               *")
        print("*********************************")
        print(" ")
    #*****************************************
    
    # Test drive the data pep:
    #*****************************************
    def test_drive_punzi_data_prep(self):
        print("Initialize data parser and data prep modules...")

        user_parser_cfg = {
            'data_loc': ['../punzinet_toolkit/sample_data/punzi_training_data.feather'],
            'file_format': 'feather'
        }
        
        this_file_path = os.path.dirname(__file__)
        default_parser_cfg_loc = os.path.join(this_file_path,"../punzinet_toolkit/cfg/defaults/pandas_parser_cfg.yaml")

        data_parser = parsers.make("PandasParser_v0",path_to_cfg=default_parser_cfg_loc,user_config=user_parser_cfg)
         
        default_prep_cfg_loc = os.path.join(this_file_path,"../punzinet_toolkit/cfg/defaults/punzi_dataprep_cfg.yaml")
        store_location = os.path.join(this_file_path,"../punzinet_toolkit/sample_data/punzi_test_dataframe")
        user_prep_cfg = {
            'loc_mass_bin_width':os.path.join(this_file_path,"../punzinet_toolkit/sample_data/belle-zpr-width.txt"),
            'df_store_loc': store_location
        }

        data_prep = preps.make("PunziDataPrep_v0",path_to_cfg=default_prep_cfg_loc,user_config=user_prep_cfg)

        print("...done!")
        print(" ")

        print("Load the data...")
        
        data = data_parser.load_data()

        # Print df, as a proof that we actually have a dataframe:
        print(data.head(5))

        print("...done!")
        print(" ")

        print("Run data prep on the dataframe...")
        
        prep_data = data_prep.run(data)

        # Print df, as a proof that we actually have a dataframe:
        print(prep_data.head(5))

        print("...done!")
        print(" ")

        if store_location is not None and store_location != "":
            print("Write dataframe to file for future tests...")
  
            data_prep.save_data(prep_data)

            print("...done!")
            print(" ")


        # Check if the weights are in the 'new' data frame:
        pass_sanity_check = False
        print("Run sanity check...")

        plt.rcParams.update({'font.size':20})
        fig, ax = plt.subplots(figsize=(12,8))

        ax.hist(prep_data['weights'].values,100)
        ax.set_xlabel('Weights added to DataFrame')
        ax.set_ylabel('Entries')
        ax.grid(True)

        fig.savefig("punzi_data_prep_weights.png")
        plt.close(fig)

        if "weights" in prep_data.columns:
            pass_sanity_check = True


        self.assertTrue(pass_sanity_check)

        print("...done! Have a great day!")
        print(" ")

    #*****************************************
        
# Run this file via: python utest_punzi_data_prep.py
if __name__ == "__main__":
    unittest.main()

        