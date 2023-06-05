# Qualitative Data Selection with Active Learning

Github project containing code written and used for our thesis-work. Includes an active learning evaluation pipeline, some object detection models and data processing.

## How to install

1. Ensure you have nvidia drivers installed.
1. Follow the detailed install information ("not quickstart") on [pip installation](https://www.tensorflow.org/install/pip) homepage.
1. Init conda for shell interaction:

    ```bash
    conda init
    ```

1. (Linux optional if installation fails) Extend temp folder:  

    ```bash
    sudo mount -o remount,size=6G /tmp
    ```

1. Use pip to install the following packages:

    ```bash
    pip install matplotlib pandas tqdm tensorflow_datasets tensorflow_addons python_dotenv
    ```

1. ImageNetV2 is currently (05-05-2023) not available through tensorflow_datasets, so it must be downloaded manually. This is hopefully fixed in the future. Download the ImageNetV2 dataset from [here](https://github.com/modestyachts/ImageNetV2)