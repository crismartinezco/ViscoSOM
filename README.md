# 🤖📚 Physics-Informed Self Organizing Maps for Viscoelastic Damping Analysis

This is the code employed in the paper "Analysis of nonlinear behavior of viscoelastic damping in musical membranes using physics-informed self-organizing maps" (
https://doi.org/10.1063/5.0242985). The entire data pipeline is explained in the paper. The code was tested with the versions of the packages included in requirements.txt.

The dataset is stored in https://zenodo.org/records/13934953.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/crismartinezco/MusicGenreClassifier](https://github.com/crismartinezco/ViscoSOM)

2. Navigate to the project directory:
   ```bash
   cd your-repo

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

The model is configured as a .py file, where the entire datapipeline happens, so please pay attention to the way the data has been annotated. These are fundamental requirements for the visualization of the SOM. There is to date no GUI since the model is very straightforward. I will add a jupyter notebook version in the future.

## Features

- Data fed to the ML is in time series - txt format, so it is necessary to transform any given audio into this format.
- FFT is run with librosa.
- Regarding the visualization of the SOM, everything is run in matplotlib. Since the shapes, colors and sizes are coupled with the data annotations, be mindful of any changes so that the chosen shapes and colors allow for easier data exploration.

## License

This project is licensed under the MIT License. Development and code was inspired from the video series Pytorch - AI for Audio from Valerio Velardo in YouTube.
