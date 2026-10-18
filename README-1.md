# CodeAlpha_HandwrittenRecognition

Machine Learning Internship @ CodeAlpha - **Task 3: Handwritten Character Recognition**

## Objective
Identify handwritten characters (digits and alphabets) using image processing and deep learning.

## Dataset
- **MNIST** - 70,000 images of handwritten digits (0-9), 28x28 grayscale
- **EMNIST Letters** (optional) - handwritten alphabets A-Z

## Approach
1. **Preprocessing:** pixel values scaled to 0-1, images shaped (28, 28, 1).
2. **Data augmentation:** random rotation, translation and zoom (training only) for robustness on real handwriting.
3. **Model (CNN):** 2 x [Conv2D -> Conv2D -> MaxPool -> Dropout] -> Flatten -> Dense(128) -> Dropout -> Softmax.
4. **Training:** Adam optimizer, early stopping, learning-rate reduction on plateau.
5. **Evaluation:** accuracy, classification report, confusion matrix, sample predictions.
6. **Prediction on new images:** `predict.py` crops, centres and resizes any image to 28x28 like MNIST.

## Project Structure
```
CodeAlpha_HandwrittenRecognition/
|-- handwritten_recognition.py   # training + evaluation
|-- predict.py                   # predict a character from an image
|-- requirements.txt
|-- README.md
|-- confusion_matrix.png         # generated after training
|-- sample_predictions.png       # generated after training
```

## How to Run
```bash
pip install -r requirements.txt

python handwritten_recognition.py                    # MNIST digits
python handwritten_recognition.py --dataset emnist   # EMNIST letters (A-Z)

python predict.py my_digit.png
```

## Input / Output
**Input:** an image of one handwritten character (e.g. `my_digit.png`)

**Output:**
```
Predicted character: 7

Top 3 predictions:
  7   99.2%
  1    0.5%
  9    0.2%
```

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | (add your result) |

![Confusion Matrix](confusion_matrix.png)
![Sample Predictions](sample_predictions.png)

## Future Work
Extend to full word/sentence recognition using CRNN (CNN + RNN + CTC loss).

## Tech Stack
Python, TensorFlow/Keras, NumPy, scikit-learn, Matplotlib, Pillow

## Author
Your Name - CodeAlpha ML Intern
