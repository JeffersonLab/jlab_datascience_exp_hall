from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

register(
    id="PunziDataPrep_v0",
    entry_point="punzinet_toolkit.data_prep.punzi_data_prep:PunziDataPrep"
)

from punzinet_toolkit.data_prep.punzi_data_prep import PunziDataPrep
