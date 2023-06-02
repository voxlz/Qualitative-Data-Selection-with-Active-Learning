# TensorFlow-Active-Learning

Project containing files written for our thesis-work. Includes active learning code and some object detection models.

## How to install

1. Follow the detailed install information ("not quickstart") on [pip installation](https://www.tensorflow.org/install/pip) homepage.
1. Nvidia drivers should be installed if on an axis computer. If not, try restarting.
1. Do steps up until step 5 (tensorflow will be installed as a dependency of axis packages)
1. Init conda for shell interaction:

    ```bash
    conda init
    ```

1. (optional) Extend temp folder:  

    ```bash
    sudo mount -o remount,size=6G /tmp
    ```

1. (optional) If you are at axis and want to use the axis model: 

    ```bash
    pip install adoctr opencv-python cta-dataloader imgaug polyaxon cta-optopus cta-optopsy paramiko scp
    ```

1. You then need to install the following packages:

    ```bash
    pip install matplotlib pandas tqdm tensorflow_datasets tensorflow_addons python_dotenv
    ```

1. Download and import required files from [here](https://axis365-my.sharepoint.com/:u:/g/personal/torbenn_axis_com/Ea6I856NK1NBl-Kl-wTTXPMBqqnnov5v2Qj_m0cA3uqjsw).
(Should include saved_models and '.env' file. Otherwise find your own weights and add the following in a new '.env')

    ```bash
    ARTIFACTORY_API_KEY=...
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    S3_ENDPOINT=s3-eu.se.axis.com
    NUM_GPUS=1 # num visible devices
    PIPELINE_BASE_DIR=./training_sessions
    ```

1. (If you did not do 8) Do change the cfg.yaml in saved models to point to LOCAL_CACHE_DIR: /mnt/build/cache_dataloader

1. Download ImageNetv2 if you want to use vgg16 from [here](https://axis365-my.sharepoint.com/:u:/g/personal/linneaal_axis_com/ETtTFcHuRKVEpMNu_B5PAkkBOuQn4HvNVoE0Ze4W8pGDIQ) and extract contents into home/tensorflow_datasets (or where you have your tensorflow_datasets folder)

1. The training pipeline requires Nvidia docker, installation instructions here: [setting up nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).

1. Make sure you have docker-compose installed. If not, do:

    ```bash
    pip uninstall docker-compose
    sudo apt-get install docker-compose-plugin
    ```

1. Finish by running:

    ```bash
    docker compose build train
    ```

1. Modify the function create_callbacks in train script to no longer expect checkpoint_path to be an input.


### Change user

sudo su -l username_here

### Attach to running process

1.      sudo apt install reptyr

1. get "pid" from nvidia-smi

1.     reptyr "pid"