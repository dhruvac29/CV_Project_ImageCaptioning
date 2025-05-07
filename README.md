# Image Captioning Project

This project implements an image-to-caption generator using a CNN-RNN architecture (ResNet50 + LSTM) to generate descriptive captions for images. The model is trained on the Flickr8k dataset.

## Project Structure

- `cv_project.ipynb`: Main Jupyter notebook containing the code for the image captioning model
- `best_model.h5`: Trained model weights file
- `features.pkl`: Pre-extracted image features from ResNet50
- `captions.txt`: Text file containing image captions from the dataset
- `model.png`: Visualization of the model architecture

## Setup

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- NLTK
- Pillow

### Installation
1. Clone this repository
2. Install the required packages:
   ```
   pip install tensorflow keras numpy pandas matplotlib nltk pillow
   ```

## Usage

1. Open the `cv_project.ipynb` notebook in Jupyter
2. Follow the steps in the notebook to:
   - Load and preprocess the data
   - Train the model (or use the pre-trained weights)
   - Generate captions for new images

## Model Architecture

The model uses a CNN-RNN architecture:
- CNN (ResNet50): Extracts features from images
- RNN (LSTM): Generates captions based on the image features

## License

This project is open source and available under the MIT License.
