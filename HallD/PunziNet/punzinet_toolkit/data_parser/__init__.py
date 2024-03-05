from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

register(
    id="PandasParser_v0",
    entry_point="punzinet_toolkit.data_parser.pandas_parser:PandasParser"
)

from punzinet_toolkit.data_parser.pandas_parser import PandasParser
