from aidapt_toolkit.registration import make, register, list_registered_modules

register(
    id="aidapt_numpy_reader_v0",
    entry_point="aidapt_toolkit.data_parsers.aidapt_numpy_reader_v0:AIDAPTNumpyReaderV0",
)
