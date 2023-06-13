# Installation Steps for Tensorflow and Horovod

1. Load the cluster specific environment

   ```bash

   # for the Rocky partition
   source ./load_env_rocky.sh
   ```

2. Install virtual environment

   ```bash
   # Example: Rocky
   export TMP_ENV_NAME="horovod-env-rocky"
   zsh ./install_venv_tensorflow_horovod.sh
   ```
