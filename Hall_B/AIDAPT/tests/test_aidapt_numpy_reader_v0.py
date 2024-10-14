import pytest
import aidapt_toolkit.data_parsers
import numpy as np
import pathlib


class TestNumpyReader:
    parser_id = "aidapt_numpy_reader_v0"
    parser_name = "test_parser"
    parser_config = {}

    rng = np.random.default_rng()

    def create_data(self):
        data_size = self.rng.integers(low=1, high=100, size=(1,))
        data = self.rng.standard_normal(size=data_size)
        return data

    def test_registration(self):
        all_parsers = aidapt_toolkit.data_parsers.list_registered_modules()
        print(all_parsers)
        assert self.parser_id in all_parsers

    def base_init(self, config):
        parser = aidapt_toolkit.data_parsers.make(self.parser_id, config = config, name = self.parser_name)
        return parser

    def test_base_init(self):
        parser = self.base_init(config = self.parser_config)

    def test_parser_get_info(self):
        parser = self.base_init(config = self.parser_config)
        output = parser.get_info()

        # Since this prints to stdout, the output should be None
        assert output is None

    def test_save_module(self, tmp_path):
        parser = self.base_init(config = self.parser_config)
        
        # Since tmp_path already exists, we should not be able to save there
        with pytest.raises(FileExistsError):
            parser.save(tmp_path)

        # Try to save the module in a new_directory
        new_dir = tmp_path.joinpath("new_directory")
        parser.save(new_dir)

        # Saving should create the directory
        assert new_dir.exists()

        # Saving should create a config.yaml file
        assert new_dir.joinpath("config.yaml").exists()

    def test_load_data(self, tmp_path):
        data = self.create_data()
        n_samples = data.shape

        data_file = tmp_path.joinpath("temp_data.npy")

        parser = self.base_init(config = {"filenames": data_file})

        parsed_data = parser.load_data()
        np.save(data_file, data)

        parser


