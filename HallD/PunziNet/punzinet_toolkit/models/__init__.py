from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

register(
    id="PunziNet_v0",
    entry_point="punzinet_toolkit.models.punzi_net:PunziNet"
)

from punzinet_toolkit.models.punzi_net import PunziNet
