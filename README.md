# AI-Based Prediction of Combinational Logic Depth

## Project Overview
Timing analysis is a crucial step in digital circuit design, but traditional synthesis-based methods are time-consuming. This project leverages **machine learning** to predict the **combinational logic depth** of signals in RTL designs, allowing engineers to identify potential timing violations without running full synthesis.

## Features
- Predicts combinational depth of signals from RTL code.
- Uses **ML models** (Random Forest, Linear Regression, Neural Networks) for predictions.
- Extracts key RTL features like **fan-in, fan-out, and gate count**.
- Reduces synthesis time significantly by providing quick estimations.

---

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **pip**
- **Git**

### Clone the Repository
```sh
git clone <your-repo-link>
cd <your-repo-folder>
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

---

## Running the Code

### 1. Prepare Dataset
Make sure you have RTL files and corresponding synthesis results. If using sample data:
```sh
python preprocess.py --data data/sample_rtl/
```

### 2. Train the Model
```sh
python train_model.py --dataset data/processed/
```

### 3. Make Predictions
```sh
python predict.py --input data/new_rtl_file.v
```

### 4. Evaluate the Model
```sh
python evaluate.py --test data/test_set/
```

---

## Repository Structure
```
├── data/                # Sample RTL files and synthesis reports
├── models/              # Trained ML models
├── src/                 # Core implementation
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── predict.py
│   ├── evaluate.py
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
```

---

## References
- Public datasets from semiconductor research papers.
- Open-source EDA tools (Yosys, ABC) for feature extraction.
- Implementation details are documented within the repository.

---

For any queries, feel free to open an **issue** in this repository!
