# Freespace Segmentation with Fully Convolutional Neural Network (FCNN)

## Build

1. In order to clone the repo to your local computer:

    ```bash
     git clone https://github.com/cakirogluozan/fo-intern-project
    ```

    or directly download from the button (but it is not recommended.)

2. (Optional) It is suggested to use virtual environment, in the root directory:
    - create a virtual environment:

        ```bash
        python3 -m venv venv
        ```

    - activate the virtual environment:

        ```bash
        source venv/bin/activate
        ```

    - deactivate the virtual environment:

        ```bash
        deactivate
        ```

    - YOU HAVE TO BE IN THE ROOT DIRECTORY FOR THIS COMMANDS.
3. Install required libraries:

    ```bash
     pip install -r requirements.txt
    ```

## Data Preprocessing

### Extracting Masks

In this part of the project, we want you to convert every JSON file into mask images:

1. Move json files into data/jsons folder.
2. Open src folder and run [`json2mask.py`](http://json2mask.py) :

    ```bash
    cd src/
    python3 json2mask.py
    ```

3. To check mask files, run `mask_on_image.py` :

    ```bash
    python3 mask_on_image.py
    ```

### Converting into Tensor

The images and masks refer to "features" and "labels" for Segmentation. In order to feed them into the Segmentation model which will be written in PyTorch, we need to format them appropriately. In this part, we will solve this issue. In the `[preprocess.py](http://preprocess.py)` there are two helper functions:

1. In order to convert images to tensor, we need  `tensorize_mask(.)` . For this complete `torchlike_data(.)`
2. In order to convert masks to tensor, we need `tensorize_mask(.)` . For this, complete `one_hot_encoder(.)`

At the end of the task, your data will be ready to train the model designed. We can use these functions!

> The above operations are mandatory for our task. In addition to these, other preprocessing techniques can be performed.

## Design Segmentation Model

There is a script to design our model: `model.py`. In this script, we could program our model. This will require research. To visualize your model at the end, you can use [this](http://alexlenail.me/NN-SVG/) website. 

## Train

We prepare [`train.py`](http://train.py) script which combines all our work. Play with hyper-parameters and examine their effects ! Enjoy 🙂