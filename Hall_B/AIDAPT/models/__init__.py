from Hall_B.AIDAPT.registration import make, register, list_registered_modules

register(
    id='tf_mlp_gan_v0',
    entry_point= "Hall_B.AIDAPT.models.tf_mlp_gan_v0:TF_MLP_GAN_V0"
)