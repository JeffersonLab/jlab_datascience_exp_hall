import punzinet_toolkit.models as models
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class UTestPunziNet(unittest.TestCase):

    # Initialize:
    #*****************************************
    def __init__(self,*args, **kwargs):
        super(UTestPunziNet,self).__init__(*args, **kwargs)

        # Get an into:
        print(" ")
        print("***************************")
        print("*                         *")
        print("*   Unit Test: Punzi Net  *")
        print("*                         *")
        print("***************************")
        print(" ")
    #*****************************************
    
    # Test drive the net:
    #*****************************************
    def test_drive_punzi_net(self):
        # Get the test data first:
        print("Load test data...")
        this_file_path = os.path.dirname(__file__)
        test_data_path = os.path.join(this_file_path,"../punzinet_toolkit/sample_data/punzi_test_dataframe.csv")

        df = pd.read_csv(test_data_path)

        print("...done!")
        print(" ")

        # Time to load the model:
        print("Load punzi net...")

        default_cfg_loc = os.path.join(this_file_path,"../punzinet_toolkit/cfg/defaults/punzi_net_cfg.yaml")
        user_cfg = {
            'model_store_loc':'punzi_net_utest_results',
            'torch_device':'cpu'
        }

        model = models.make("PunziNet_v0",path_to_cfg=default_cfg_loc,user_config=user_cfg)

        print("...done!")
        print(" ")

        # Run a small training:
        print("Run short training cycle...")

        history = model.train(df)

        print("...done!")
        print(" ")
    #*****************************************

# Run this file via: python utest_punzi_net.py
if __name__ == "__main__":
    unittest.main()
