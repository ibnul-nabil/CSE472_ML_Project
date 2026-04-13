# CSE472 ML Project: Bengali Audio Deepfake Generation 

## Overview
This project is focused on the processing, generation, and evaluation of bengali audio deepfakes. It provides tools for preparing datasets, training and fine-tuning models, generating synthetic audio, and evaluating the quality of generated audio.<br>
Kaggle Notebook : https://www.kaggle.com/code/tausifr/ml-project-4-2 <br>
Trained Models : https://huggingface.co/tsfrsd/vits_voice_clone/tree/main <br>
Deepfake Samples : https://huggingface.co/datasets/tsfrsd/bangla_deepfake_samples

---

## Project Structure

- **Audio_Processing/**
  - Scripts and data for audio preprocessing, transcription, and dataset generation.
  - Handles splitting, cleaning, and preparing audio files and metadata for model training.

- **Generation/**
  - Contains workflows for model fine-tuning, inference, and evaluation.
  - Includes pre-trained and fine-tuned model checkpoints, configuration files, and Jupyter notebooks for training/testing.

- **test.py / ecapaTest.py / ecapa_test.ipynb**
  - Scripts and notebooks for evaluating the quality of generated (deepfake) audio.
  - Used for testing model outputs and benchmarking against real audio.

---

## Getting Started

1. **Audio Processing**
   - Use scripts in `Audio_Processing/` to preprocess raw audio, transcribe speech, and generate training datasets.
   - Example: Run `split_script.py` or `transcribe_kaggle.py` as needed.

2. **Model Training & Generation**
   - Use notebooks and scripts in `Generation/` to fine-tune models or run inference.
   - Example: Use `train_kaggle.py` or `training.ipynb` for training; use `test_best_model_tts.ipynb` for inference/testing.

3. **Evaluation**
   - Use `test.py`, `ecapaTest.py`, or `ecapa_test.ipynb` in the project root to evaluate the quality of generated audio.

---

## Requirements
- Python 3.x
- Common ML/DL libraries (PyTorch, librosa, etc.)
- See `Kaggle_installs.txt` in `Generation/` for a list of required packages.

---

## Notes
- See individual script and notebook files for detailed usage instructions.
- Model checkpoints and large datasets are not included in version control.

---

## Acknowledgement
- The base VITS model, training script and part of the dataset was taken from the Banglafake dataset <br>
https://huggingface.co/sifat1221/vits_bn_tts_checkpoint_140000<br>
https://huggingface.co/datasets/sifat1221/banglaFake

