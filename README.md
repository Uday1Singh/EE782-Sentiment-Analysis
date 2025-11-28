# Speech Emotion Recognition using Wav2Vec2 & Attention Pooling

A robust Deep Learning system capable of classifying human speech into 8 distinct emotions with **87.50% accuracy**. This project leverages **Self-Supervised Learning (SSL)** by fine-tuning the **Wav2Vec2** transformer model and enhancing it with a custom **Attention Pooling** mechanism.

## 📋 Abstract

Speech Emotion Recognition (SER) is challenging due to the gap between raw audio signals and subjective emotions. Traditional methods (MFCCs + CNNs) often fail to capture long-range temporal dependencies.

This project implements a transfer learning approach using Facebook's **Wav2Vec2-base**. By unfreezing the transformer encoder layers and attaching a learnable Attention Pooling head, the model effectively predicts emotions from the **RAVDESS** dataset.

### Key Features

  * **Transformer Backbone:** Uses pre-trained `wav2vec2-base` for contextualized speech representations.
  * **Attention Pooling:** A custom pooling layer weighting audio frames by emotional salience.
  * **High Accuracy:** Achieves **87.50%** test accuracy, beating standard CNN baselines (\~74%).
  * **Real-Time Inference:** Includes a Tkinter-based GUI for live microphone predictions.

## 📊 Dataset

Trained on the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

  * **Classes (8):** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgusted, Surprised.
  * **Preprocessing:** Resampled to **16 kHz**, normalized, fixed to **5 seconds**.

## 🛠️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/speech-emotion-recognition.git
    cd speech-emotion-recognition
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1\. Training

To train the model (or fine-tune):

1.  Download RAVDESS dataset to `data/`.
2.  Run training:
    ```python
    python main.py --mode train
    ```
    *Note: GPU recommended.*

### 2\. Inference (GUI)

Test with your own voice:

```bash
python main.py --mode gui
```

  * Click **"Record"**, speak for 5s, and get the predicted emotion.

## 🧠 Architecture & Results

| Component | Description |
| :--- | :--- |
| **Feature Extractor** | 7-layer CNN (Frozen Wav2Vec2 weights) |
| **Encoder** | 12-layer Transformer (Unfrozen/Fine-tuned) |
| **Pooling** | **Attention Pooling**: $\sum \alpha_t h_t$ |
| **Classifier** | MLP: Dense(256) $\to$ Dense(128) $\to$ Output(8) |

**Performance:**

  * **Test Accuracy:** **87.50%** (15 epochs, AdamW, lr=$3e^{-5}$)
  * **F1-Score:** Excels at high-energy emotions (Angry: 0.91, Disgusted: 0.93).

## 👥 Team Members

Developed for **EE782: Advanced Machine Learning**:

  * **Uday Singh** (22B1262)
  * **Ankit Maurya** (22B1266)
  * **Aditya Bhadoria** (22B1247)
