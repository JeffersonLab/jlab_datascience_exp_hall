from aidapt_toolkit.registration import make, register, list_registered_modules

register(
    id='tf_mlp_gan_v0',
    entry_point= "aidapt_toolkit.models.tf_mlp_gan_v0:TF_MLP_GAN_V0"
)

register(
    id='tf_cgan_v0',
    entry_point= "aidapt_toolkit.models.tf_cgan_v0:TF_CGAN"
)