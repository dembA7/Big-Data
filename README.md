# Fashion MNIST Classification with PySpark

## Introduction
This project aims to build a classification model using the Fashion MNIST dataset, leveraging the power of PySpark for data processing and machine learning. The Fashion MNIST dataset contains images of clothing items, which are labeled with categories such as T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.

## Project Structura
```bash
Big-Data/
│
├── data/                      # Directory containing datasets
│   ├── fashion-mnist_train.csv
│   └── fashion-mnist_test.csv
│
├── models/                    # Directory for storing trained models
│
├── data_preparation.py        # Script for preparing the dataset
├── model_training.py          # Script for training the classification model
├── model_evaluator.py         # Script for evaluating the model's performance
├── model_metrics.txt          # File for storing model metrics
└── predictions/               # Directory containing predictions
```

## Getting Started
**Prerequisites**
- Python 3.x
- PySpark
- Git
- WSL (Windows Subsystem for Linux)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/dembA7/Big-Data.git
cd Big-Data
```

2. Set up a virtual environment (optional but recommended): 
```bash
python -m venv pyspark-env
source pyspark-env/bin/activate  # For Linux
```

3. Install required packages:
```bash
pip install pyspark gitpython opencv-python matplotlib pandas
```

## Usage
1. Prepare the data: Run the data_preparation.py script to preprocess the Fashion MNIST dataset.
```bash
python data_preparation.py
```

2. Train the model: Execute the model_training.py script to train the classification model.
```bash
python model_training.py
```

3. Evaluate the model: Run the model_evaluator.py script to evaluate the trained model and save predictions.
```bash
python model_evaluator.py
```

## Dataset

The **Fashion MNIST** dataset is a popular benchmark dataset for machine learning, consisting of 70,000 grayscale images of clothing items. Each image is 28x28 pixels and belongs to one of 10 categories:

1. **T-shirt/top**
2. **Trouser**
3. **Pullover**
4. **Dress**
5. **Coat**
6. **Sandal**
7. **Shirt**
8. **Sneaker**
9. **Bag**
10. **Ankle boot**

The dataset is split into two parts: 

- **Training set**: 60,000 images used to train the model.
- **Test set**: 10,000 images used to evaluate the model's performance.

Each category is labeled with a corresponding integer from 0 to 9, facilitating supervised learning. This dataset serves as a great introduction to image classification and is commonly used for benchmarking classification algorithms.

## Conclusion
This project demonstrates the capability of using PySpark for handling large datasets and building machine learning models efficiently. The Fashion MNIST dataset serves as an excellent benchmark for classification tasks.

## License
This project is licensed under the GNU General Public License. See the [LICENSE](LICENSE) file for details.