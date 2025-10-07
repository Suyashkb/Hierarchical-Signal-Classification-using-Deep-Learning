#  Hierarchical Deep Learning System for Radio Signal Classification

This repository presents a **hierarchical deep learning system** for **Automatic Modulation Classification (AMC)**, capable of identifying one of **11 modulation types** from raw I/Q radio signals.  
The system leverages three specialized neural network models — **Binary CNN**, **Non-Phase CNN-LSTM**, and **Phase-based CNN** — arranged in a pipeline to maximize accuracy and efficiency.

---

##  Model Summary

This is a **multi-model hierarchical classifier** designed for robust signal modulation recognition.  

The system operates in three stages:
1. **Binary Model (CNN):**  
   Determines whether the input signal uses *phase-based* or *non-phase-based* modulation.
2. **Phase Model (CNN):**  
   Classifies phase-based modulations using constellation plot images.
3. **Non-Phase Model (CNN-LSTM):**  
   Handles temporal features of non-phase modulations directly from I/Q data.

This **divide-and-conquer** architecture allows each model to specialize in its domain, leading to improved overall classification performance.

---

##  Usage

This AMC system can be applied in:
- Cognitive Radio Systems  
- Spectrum Monitoring  
- Interference and Signal Detection  
- Wireless Research and Education  

>  *Primary failure mode:*  
> A misclassification by the Binary model (rare, as it achieves >99% accuracy).  
> Performance degrades under very low SNR (Signal-to-Noise Ratio) conditions.

---

##  Full Prediction Pipeline

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define class labels
NON_PHASE_CLASSES = ['PAM4', 'GFSK', 'CPFSK', 'WBFM', 'AM-DSB', 'AM-SSB']
PHASE_CLASSES = ['QPSK', 'QAM16', 'QAM64', 'BPSK', '8PSK']

# Load trained models
binary_model = tf.keras.models.load_model('best_binary_model.keras')
non_phase_model = tf.keras.models.load_model('best_model_non_phase.keras')
phase_model = tf.keras.models.load_model('best_model.keras')

def create_constellation_image(signal_frame):
    I, Q = signal_frame[:, 0], signal_frame[:, 1]
    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
    ax.plot(I, Q, '.')
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    img_gray = Image.fromarray(img).convert('L')
    img_resized = img_gray.resize((192, 256))  # W, H
    img_array = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=[0, -1])

def predict_modulation(signal):
    binary_input = np.expand_dims(signal, axis=[0, -1])
    is_phase_prob = binary_model.predict(binary_input)[0][0]

    if is_phase_prob > 0.5:
        # Phase-based path
        constellation_img = create_constellation_image(signal)
        prediction = phase_model.predict(constellation_img)
        return PHASE_CLASSES[np.argmax(prediction)]
    else:
        # Non-phase path
        non_phase_input = np.expand_dims(signal, axis=[0, 1])
        prediction = non_phase_model.predict(non_phase_input)
        return NON_PHASE_CLASSES[np.argmax(prediction)]
```
# Example
dummy_signal = np.random.randn(128, 2)
predicted_class = predict_modulation(dummy_signal)
print(f"Predicted Modulation Class: {predicted_class}")

##  System Overview

- **Input:** Raw In-phase and Quadrature (I/Q) signal, shape `(128, 2)`  
- **Output:** Predicted modulation class *(string)*, e.g., `'QPSK'`  
- **Pipeline:**  
  `Binary CNN → Phase CNN or Non-Phase CNN-LSTM → Final Class`  
- **Dependencies:** Python, TensorFlow, NumPy, Matplotlib, Pillow  

---

##  Implementation Details

| Component | Description |
|------------|-------------|
| **Software** | Python 3.8+, TensorFlow (Keras API) |
| **Hardware (Training)** | GPU (NVIDIA V100 / A100) or TPU |
| **Hardware (Inference)** | Works efficiently on CPU or GPU |
| **Training Duration** | Few hours to a day (depends on hardware) |
| **Inference Time** | < 1 second per signal (CPU), faster on GPU |

---

##  Model Characteristics

| Model | Type | Description |
|--------|------|-------------|
| **Binary Model** | CNN | 2 Conv2D + 2 Dense layers; fast and accurate (>99%) |
| **Non-Phase Model** | CNN-LSTM | 3 Conv2D + 2 LSTM layers; processes temporal patterns |
| **Phase Model** | CNN | 3 Conv2D + 1 Dense layer; classifies constellation images |

###  Initialization

All models were **trained from scratch** using Keras’ default **Glorot uniform** initialization.  
No transfer learning or pre-trained weights were used.

---

##  Dataset

- **Dataset:** RML22 *(based on RadioML 2016.10a)*  
- **Source:** Synthetic wireless signals generated using GNU Radio
- **Classes (11 total):** ['BPSK', 'QPSK', '8PSK', 'PAM4', 'QAM16', 'QAM64','GFSK', 'CPFSK', 'WBFM', 'AM-DSB', 'AM-SSB']
- **SNR Range:** -20 dB to +18 dB  
- **Total Samples:** 462,000 *(2,000 signals per modulation-SNR pair)*  
- **Train/Validation Split:** 80/20  

---

##  Evaluation Results

| Model | Accuracy | Notes |
|--------|-----------|-------|
| **Binary Model** | >99% | Nearly perfect separation of phase/non-phase |
| **Non-Phase Model** | ~80% | PAM4: 75%, GFSK: 85%, CPFSK: 78%, WBFM: 64%, AM-DSB: 82%, AM-SSB: 85% |
| **Phase Model** | High | Vision-based constellation classification |
| **Overall System** | Robust | Hierarchical integration yields strong accuracy |

---

##  Fairness & Ethics

- **Fairness:**  
Balanced class distribution, using `sparse_categorical_crossentropy` ensures no class bias.

- **Usage Limitations:**  
- Not reliable at extremely low SNRs.  
- Trained only on 11 modulation types *(won’t recognize unseen ones)*.

- **Ethical Considerations:**  
This model is intended for **academic and research purposes** such as spectrum management.  
Misuse in **unauthorized surveillance or military applications** is **strongly discouraged**.

---

##  Data Source

**Dataset:** *RML22: Realistic Dataset Generation for Wireless Modulation Classification*  
**Reference Paper:**  
> V. Sathyanarayanan, P. Gerstoft, and A. E. Gamal,  
> *"RML22: Realistic Dataset Generation for Wireless Modulation Classification,"*  
> IEEE Transactions on Wireless Communications, vol. 22, no. 11, pp. 7663–7675, Nov. 2023.  
> DOI: [10.1109/TWC.2023.3254490](https://doi.org/10.1109/TWC.2023.3254490)

---

##  Author

**Suyash Kumar Bhagat**  
B.Tech, Electronics and Communication Engineering  
Delhi Technological University (DTU)  
📧 [suyashkbhagat_ec22a18_53@dtu.ac.in](mailto:suyashkbhagat_ec22a18_53@dtu.ac.in)

---

##  Kaggle Model

Explore the full implementation and documentation on Kaggle:  
-> [Kaggle Notebook (Signal Classification Models)](https://www.kaggle.com/models/suyashkumarbhagat/signal-classification-models) 

---

##  License

This project is released under the **MIT License**.

---


