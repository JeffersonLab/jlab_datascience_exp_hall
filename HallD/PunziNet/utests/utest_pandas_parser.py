import punzinet_toolkit.data_parser as parsers
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class UTestPandasParser(unittest.TestCase):

    # Initialize:
    #*****************************************
    def __init__(self,*args, **kwargs):
        super(UTestPandasParser,self).__init__(*args, **kwargs)

        # Get an into:
        print(" ")
        print("********************************")
        print("*                              *")
        print("*   Unit Test: Pandas Parser   *")
        print("*                              *")
        print("********************************")
        print(" ")
    #*****************************************
        
    # Test drive the parser:
    #*****************************************
    def test_drive_pandas_parser(self):
        print("Create test data set(s)...")

        # Specify number of files and events you wish to generate:
        n_data_files = 10
        n_events_per_file = 1000
        data_locs = []

        # Settings for the test data itself:
        neg_loc = -1.0
        pos_loc = 1.0
        #+++++++++++++++++++++++
        for i in range(n_data_files):
            output_loc = 'test_data' + str(i) + '.csv'

            # Create dictionary:
            test_data = {
              'a':np.random.normal(loc=neg_loc,scale=0.5,size=(n_events_per_file,)),
              'b':np.random.normal(loc=pos_loc,scale=0.5,size=(n_events_per_file,))
            }

            # Turn everything into a dataframe:
            df = pd.DataFrame(test_data)

            # And write it to file:
            df.to_csv(output_loc)
            
            # Record file names, so we can test our parser:
            data_locs.append(output_loc)
        #+++++++++++++++++++++++
        
        
        print("...done!")
        print(" ")

        # Load the data parser:
        print("Load pandas parser...")
        
        # Get path to the default configs, so we have less to specify:
        this_file_path = os.path.dirname(__file__)
        default_cfg_loc = os.path.join(this_file_path,"../punzinet_toolkit/cfg/defaults/pandas_parser_cfg.yaml")
        
        # Here, we need a user config, because we do not know that path to each data file beforehand:
        user_cfg = {
            'data_loc': data_locs
        }

        # Get the parser:
        pandas_parser = parsers.make("PandasParser_v0",path_to_cfg=default_cfg_loc,user_config=user_cfg)

        # Print the info:
        pandas_parser.get_info()

        print("...done!")
        print(" ")

        # Load the data:
        print("Load data...")

        data = pandas_parser.load_data()

        # Print df, as a proof that we actually have a dataframe:
        print(data.head(5))

        print("...done!")
        print(" ")

        # Run a dimensional check. If everything went right, we should end up with a single dataframe
        # that has n_files*n_events_per_file events
        print("Run dimensional check...")

        passDimensionCheck = False
        if data.shape[0] == n_data_files*n_events_per_file:
            passDimensionCheck = True

        print("...done!")
        print(" ")

        # Plot the test data:
        print("Visualize data...")

        plt.rcParams.update({'font.size':20})
        fig, ax = plt.subplots(1,2,figsize=(15,8),sharey=True)

        n0,_,_ = ax[0].hist(data['a'],100,label='Data')
        n_max = np.max(n0)
        ax[0].plot([neg_loc,neg_loc],[0,n_max],'r--',linewidth=3.0,label='Expected Mean')
        ax[0].set_xlabel('Column a')
        ax[0].set_ylabel('Entries')
        ax[0].legend()
        ax[0].grid(True)

        ax[1].hist(data['b'],100,label='Data')
        ax[1].plot([pos_loc,pos_loc],[0,n_max],'r--',linewidth=3.0,label='Expected Mean')
        ax[1].set_xlabel('Column b')
        ax[1].legend()
        ax[1].grid(True)

        fig.savefig('pandas_parser_test_data.png')
        plt.close(fig)

        print("...done!")
        print(" ")

        # Clean up everything:
        print("Remove test data set(s)...")

        #+++++++++++++++++++
        for data_set in data_locs:
          os.remove(data_set)
        #+++++++++++++++++++

        print("...done!")
        print(" ")

        # Check that we passed the dimension test:
        self.assertTrue(passDimensionCheck)

        print("Have a great day!")
    #*****************************************
        

# Run this file via: python utest_pandas_parser.py
if __name__ == "__main__":
    unittest.main()
