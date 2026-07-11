# InternGrow_AdvancedOCREngine

**InternGrow Machine Learning Track — Task 3**

## 📌 Objective
Train a deep CNN to classify handwritten digits/characters.

## 🧠 Approach
- Dataset: scikit-learn's `load_digits` (1,797 8x8 grayscale handwritten digit images, from the same UCI/NIST family as MNIST) — used so the project runs fully offline with no large download required
- Model: CNN (Conv2D + BatchNorm + MaxPooling + Dropout) built with TensorFlow/Keras
- Achieved **~99% test accuracy**

## 📂 Files
- `ocr_engine.py` — full pipeline: data loading, CNN model, training, evaluation, plotting
- `sample_digits.png`, `training_history.png`, `confusion_matrix.png` — generated after running
- `handwritten_digit_cnn.keras` — saved trained model

## ▶️ How to Run
```bash
pip install numpy scikit-learn matplotlib tensorflow
python ocr_engine.py
```

## 📈 Results (example run)
- **Test Accuracy: 98.9%**
- **Test Loss: 0.027**

## 🔁 Upgrade Feature — Full Word Segmentation Pipeline
The task's upgrade feature calls for segmenting full handwritten words from an uploaded image into editable digital text. This is a substantial follow-on project on top of the core digit/character classifier here:
1. **Preprocessing**: binarize + denoise the uploaded image (OpenCV)
2. **Segmentation**: contour detection to isolate individual characters/words (`cv2.findContours`), or a CRNN (CNN + BiLSTM + CTC loss) for end-to-end sequence recognition without explicit segmentation
3. **Recognition**: feed each segmented character through this CNN (retrained on EMNIST for full alphanumeric support)
4. **Output**: reassemble predicted characters into an editable text string

This is a good "Phase 2" to build once the core classifier (in this repo) is validated, and is noted here as a roadmap rather than included in this submission due to time constraints.

## 🔁 Scaling Up to Full MNIST / EMNIST
To use the full 28x28 MNIST dataset or EMNIST (letters) instead of `load_digits`:
```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
```
The same CNN architecture (input_shape adjusted to `(28, 28, 1)`) works unchanged.

## 🎥 Submission Checklist (InternGrow)
- [ ] Push to GitHub as `InternGrow_AdvancedOCREngine`
- [ ] Record project video, post on LinkedIn tagging @InternGrow, with GitHub link
- [ ] Submit via the InternGrow submission form
