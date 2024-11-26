# Multi-Label-Image-Classification
## Features

> **Code Decoupling:** Decouple all data loading, network model construction, model training and validation
>
> **Rich Content:** Providing rich evaluation indicators and functional functions

## Function Functionality

> - checkpoints: stores the weights of the trained model;
> - datasets: Store datasets. And partition the dataset;
> - logs: Stores training logs. Including losses and accuracy during training and validation;
> - option. py: stores all the parameters required for the entire project;
> - utils.py: Stores various functions. Including model saving, model loading, and loss function, etc;
> - split_data. py: Divide the dataset;
> - model. py: Building a neural network model;
> - train.py: Train the model;
> - predict. py: Evaluate the training model;
> - model_transfer. py: Transfer .pth model to .onnx model.

## Requirements

Required:



> matplotlib==3.8.3
> numpy==1.26.4
> Pillow==9.5.0
> Pillow==10.3.0
> scikit_learn==1.5.0
> torch==2.2.1
> torchvision==0.17.1
> tqdm==4.66.2



You can install these dependencies via pip:

```python
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

Prepare your dataset and place it according to the following requirements:

> ```python
> datasets
>    images
>        styles.csv
>    ```



If you need to conduct code validity testing first, you can use  [Fashion-Product-Images-Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) first.

### 2. Split Your Dataset

Run the following script to obtain the partitioned dataset, but you need to pay attention to modifying some paths:

```python
python split_data.py
```

Then you can obtain the following data structure:

> ```python
> datasets
>    images
>        styles.csv
>        train.csv
>        val.csv
>  ```

### 3. Modify Network 

How many categories do you need to predict? You need to add a few category headers in **model.py**.

### 4. Train Your Dataset

Run the following script to train your dataset and output various parameters during the training and validation processes:



```python
python train.py
```

### 5. Evaluate Your Model

Running the following code can evaluate the accuracy of your model. You can choose to evaluate the test dataset or individual images:



```python
python predict.py
```

## License

This project is licensed under the Apache 2.0 license. For detailed information, please refer to the LICENSE file.
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Acknowledgement

> Kaggle Dataset: [Fashion-Product-Images-Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)



> CSDN: [王乐予-CSDN博客](https://blog.csdn.net/qq_42856191?type=blog)
























