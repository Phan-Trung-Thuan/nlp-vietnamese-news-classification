# Vietnamese News Classification using NLP and KAN-Linear

This project explores various deep learning architectures for classifying Vietnamese news articles into 12 categories. It leverages the Kolmogorov-Arnold Network (KAN) Linear layer as a classifier on top of common backbones.

## Dataset
The dataset used for this project consists of Vietnamese news articles and can be found on Kaggle:
[Vietnamese Newspaper Dataset](https://www.kaggle.com/datasets/phantrungthuan/vietnamese-newspaper-dataset)

The preprocessing includes stopword removal using the [vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords) library.

## Project Structure

```
├── data/               # Dataset directory (raw and processed)
├── docs/               # Project report and presentation
├── notebooks/          # Original Jupyter notebooks for exploration
├── src/                # Source code
│   ├── data/           # Data loading and preprocessing logic
│   ├── models/         # Model architectures (CNN, LSTM, GRU, MiniBERT)
│   ├── utils/          # Training, evaluation, and inference utilities
│   ├── config.py       # Configuration and argument parsing
│   ├── train.py        # Main training script
│   ├── eval.py         # Evaluation script
│   └── infer.py        # Inference script
├── weights/            # Pre-trained model weights
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Architectures

The following models are implemented:
- **CNN1D**: 1D Convolutional Neural Network with KAN-Linear classifier.
- **BiLSTM**: Bidirectional Long Short-Term Memory network.
- **BiGRU**: Bidirectional Gated Recurrent Unit network.
- **MiniBERT**: A lightweight Transformer-based architecture using Linear Attention.

All models use a `KANLinear` layer instead of a traditional Fully Connected layer for classification, due to its strong non-linear mapping capabilities.

## Experiment Results

### Hyperparameters
| Hyperparameter | 1D CNN | BiLSTM | BiGRU | MiniBERT |
| --- | --- | --- | --- | --- |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW |
| **Learning rate** | 0.005 | 0.005 | 0.005 | 0.005 |
| **Weight decay** | 0.01 | 0.01 | 0.01 | 0.01 |
| **Batch size** | 64 | 64 | 64 | 64 |
| **Num epochs** | 50 | 50 | 50 | 50 |

### Metrics
| Metrics | 1D CNN | BiLSTM | BiGRU | MiniBERT |
| --- | --- | --- | --- | --- |
| **Accuracy** | 0.61634 | 0.42539 | 0.68047 | 0.69417 |
| **F1 score (Weighted)** | 0.61809 | 0.43237 | 0.68302 | 0.69359 |

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd news-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To train a model from scratch, use the `train.py` script. You can specify the model architecture using the `--model` flag (`1d_cnn`, `bi_lstm`, `bi_gru`, `minibert`):

```bash
cd src
python train.py --model minibert --epochs 50 --batch_size 64
```

### Evaluation
To evaluate a trained model on the test dataset:

```bash
cd src
python eval.py --model minibert
```

### Inference
To run inference on a specific text string:

```bash
cd src
python infer.py --model minibert --text "Thủ tướng nhấn mạnh chuyển đổi số là xu thế tất yếu..."
```

## Credits
- KAN-Linear implementation adapted from [efficient-kan](https://github.com/Blealtan/efficient-kan).
- Stopwords from [vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords).
