# Upscaling EEG Data

A neural network autoencoder that upscales 8-electrode EEG signals to 16-electrode signals, effectively synthesizing the missing electrode readings from a lower-density cap.

## Approach

- Autoencoder architecture built with TensorFlow/Keras
- Input: 8-channel EEG trial data
- Output: reconstructed 16-channel signal
- Data preprocessing with `extract_eeg.py`; training/testing splits stored as `.npy` files

## Usage

```bash
pip install -r requirements.txt
python neural_autoencoder_M.py
```

Set `train_autoencoder = True` in `neural_autoencoder_M.py` to train from scratch, or `False` to load a saved model.
