# 🎙️ Voice‑Based Gender Classification from Speech

This project is about building a **voice‑based gender classification system** using real‑world speech data. The goal is to predict whether a given voice sample belongs to a **male or female speaker** by learning patterns from audio signals.

The entire pipeline — from raw audio files to model prediction — is implemented step by step. The dataset used comes from the **Mozilla Common Voice Project (Tamil language)**, which makes this project practical and close to real‑world scenarios.

---

## 📌 What this project does

* Takes raw speech audio as input
* Extracts meaningful audio features (MFCCs)
* Trains deep learning models on those features
* Predicts the speaker’s gender for unseen audio samples

Two different models are implemented and compared to understand how model choice impacts performance.

---

## 📂 Dataset Information

* **Source:** Mozilla Common Voice (Open‑source speech dataset)
* **Language:** Tamil (`ta`)
* **Audio format:** `.mp3`
* **Gender labels used:**

  * `male_masculine`
  * `female_feminine`

Initially, the dataset was imbalanced. To avoid biased learning, the data was **balanced using undersampling**, so both genders have an equal number of samples.

### Dataset after balancing

| Gender    | Number of Samples |
| --------- | ----------------- |
| Male      | 5605              |
| Female    | 5605              |
| **Total** | **11210**         |

---

## ⚙️ Workflow Overview

1. Extract and load the Common Voice dataset
2. Filter valid gender labels
3. Balance the dataset
4. Organize audio files by gender
5. Extract MFCC features from each audio clip
6. Train deep learning models
7. Evaluate and compare results
8. Predict gender for new voice samples

---

## 🔊 Feature Extraction (MFCC)

To represent audio numerically, **Mel‑Frequency Cepstral Coefficients (MFCCs)** are used. MFCCs capture important characteristics of human speech and are widely used in speech and audio processing tasks.

* Library: `librosa`
* MFCCs extracted per file: `13`
* Final feature: mean of MFCC values over time

```python
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfcc.T, axis=0)
```

---

## 🧠 Models Used

### 1️⃣ Feed Forward Neural Network (Baseline Model)

This model serves as a **baseline** to understand how well simple dense layers perform with MFCC features.

**Architecture:**

* Dense (128) → ReLU
* Dropout (0.5)
* Dense (64) → ReLU
* Dropout (0.5)
* Dense (2) → Softmax

**Results:**

* Test Accuracy: **97.9%**

This model already gives strong performance, showing that MFCC features are highly informative for this task.

---

### 2️⃣ Recurrent Neural Network (Bi‑Directional LSTM)

To capture temporal relationships within MFCC features, a **Bi‑Directional LSTM** model is used. This model learns patterns from both forward and backward directions.

**Architecture:**

* Bi‑Directional LSTM (128 units)
* Dropout (0.5)
* Dense (64) → ReLU
* Dropout (0.5)
* Dense (1) → Sigmoid

**Results:**

* Test Accuracy: **98.75%**

The LSTM model slightly outperforms the baseline network, making it the better choice for this problem.

---

## 📊 Observations

* Both models perform very well on unseen data
* LSTM model achieves higher accuracy and better generalization
* Dropout layers help reduce overfitting
* MFCC features are effective for voice‑based gender classification

---

## 🔮 Gender Prediction

The trained model can be used to predict gender for any new audio file:

1. Extract MFCC features from the audio
2. Pass features into the trained model
3. Get gender prediction along with probability

**Sample output:**

```
Result: Male
Male Probability: 88.49%
Female Probability: 11.51%
```

---

## 🛠️ Tools & Technologies

* Python
* TensorFlow / Keras
* Librosa
* NumPy
* Pandas
* Scikit‑learn
* Google Colab

---

## 📁 Project Structure

```
├── voice_dataset.ipynb
├── mfcc_data_df.csv
├── male_clips/
├── female_clips/
├── gender_detection_model.h5
├── gender_rnn_model.h5
└── README.md
```

---

## 🚀 Future Scope

* Use full MFCC sequences instead of averaged features
* Experiment with CNN or CNN‑LSTM models
* Extend support to more languages
* Deploy as a web or mobile application

---

##  Acknowledgements

* **Mozilla Common Voice Project** for providing free and open‑source speech data
* Open‑source community for libraries and tools used

---

