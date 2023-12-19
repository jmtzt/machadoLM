# Machado de Assis - Transformer-based Language Model

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset
The dataset used in this project is a collection of the books written by Machado de Assis, a Brazilian writer. The dataset was downloaded from Kaggle and can be found [here](https://www.kaggle.com/datasets/luxedo/machado-de-assis)

After downloading the dataset, unzip it in the `data` folder then run the following command to prepare the data into a single `.txt` file that will be used to train our transformer model.

```bash
python data.py
```

## Training

To train the simple bigram model, run the following command:

```bash
python train_bigram.py
```

To train the transformer model, run the following command:

```bash
python train.py
```

The weights of the transformer model will be saved in the `weights` folder.
In total, this particular model takes about 15 minutes to train on a 4090.

