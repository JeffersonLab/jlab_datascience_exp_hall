# Hall B - AIDAPT Workflow

### Running the Inner GAN Driver
- Initialize environment with necessary packages (tensorflow, numpy, matplotlib)
- Ensure jlab_datascience_core has been installed
  - Core repo link: https://github.com/JeffersonLab/jlab_datascience_core
  - Use local editable pip install (`pip install -e .` from the core repo directory)
- Install the aidapt_toolkit (from the AIDAPT directory)
- From the AIDAPT directory, execute:
`python3 ./aidapt_toolkit/drivers/basic_inner_gan_driver.py ./aidapt_toolkit/configs/basic_driver_config.yaml`
