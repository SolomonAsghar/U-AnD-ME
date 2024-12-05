[![LinkedIn][linkedin-shield]][linkedin-url]

<div align="center">
  <h1 align="center">U-AnD-ME</h1>
  U-Net 3+ for Anomalous Diffusion analysis enhanced with Mixture Estimates. 
</div>
<br />
This repo contains code for the method that team "UCL SAM" used for the AnDi Challenge 2024. We came 1st place in both trajectory analysis tasks. We also came 1st for each of the five subtasks of the single trajectory task, which was the most heavily subscribed across the contest.

#### Overview:
Our method uses a neural network to make predictions on a per-timestep basis for each trajectory. These timestep level predictions can be processed into segment level predictions. Additionally, timestep predictions from all trajectories across an experiment are combined to predict which phenomenological model corresponds to that experiment, and to generate a Gaussian mixture model (GMM) quantifying the experimental ensemble’s dynamic properties. To further increase prediction accuracy, experiment specific networks are created, each trained on trajectories resembling those from their experiment, i.e. trajectories from the predicted model generated with properties following the generated GMM.

#### Network Architecture:
Our network architecture is inspired by UNet 3+. We use a deep encoder-decoder composed of 1D convolutions and transposed convolutions, with full-scale skip connections. The network accepts 224 × 2 (timesteps × dimensions) matrices as input. Several outputs are generated for each timestep: a sigmoid output predicts the probability of the timestep being a changepoint, a 5-way softmax predicts the model, and finally α, K and diffusion type are each simply given by linear outputs. Experiment specific networks do not output any model predictions.
 
#### Training:
The andi-datasets package is used to generate trajectories for training. Initially, trajectories corresponding to every model are generated, with all their parameters being randomly sampled from predefined ranges. When experiment specific networks are trained, only trajectories from models deemed to be likely are generated, with α and K sampled from the experiment’s generated GMM.

We process each trajectory into a training sample by differencing along the time axis and padding to a length of 224 timesteps. Samples are padded with zeros; their corresponding labels are padded with zeros apart from diffusion type, which we assign as “transient-confinement model”. In a sense, our padding method treats the boundary of every FOV as an immobilizing trap.

Once 50,000 training samples are generated, training proceeds until the validation loss stagnates for 3 consecutive epochs. Then, new training data is generated and another training iteration occurs. Training comes to a final stop when the validation loss stagnates for 3 consecutive training iterations.

#### Ensemble predictions:
Following the necessary padding and differencing preprocessing steps, predictions are made for all the trajectories in an experiment. Then, padding is reversed to remove outputs corresponding to timesteps not in the original trajectories. The model is predicted by averaging over all softmax outputs and seeing which model was assigned the highest probability. A GMM is generated using each timestep’s assigned α, K and diffusion type. The number of GMM components is decided by creating GMMs with a range of components, from 1 to 10, and seeing which has the lowest Bayesian information criterion.

#### Single Trajectory Predictions:
Outputs are split according to their predicted change points to generate segments. The values of α, K and diffusion type for each timestep across a segment are averaged to generate a singular prediction for the segment. This average uses a parabolic weighting, where timesteps near the centre of the segment contribute to the average more than those at its extremities.  

# Software Requirements
### OS Requirements
This code is compatible with Windows and Unix operating systems. It has been tested on Rocky Linux 8.

### Dependencies
U-AnD-ME runs using Python 3.10 with TensorFlow 2.10. All the libraries used are:

[![NumPy][NumPy-badge]][NumPy-url]

[![SKLearn][SKLearn-badge]][SKLearn-url]

[![SciPy][SciPy-badge]][SciPy-url]

[![TensorFlow][TensorFlow-badge]][TensorFlow-url]


# Installation Guide
Once the required libraries are installed, simply clone the repository with Git LFS:
```
git lfs clone https://github.com/SolomonAsghar/U-AnD-ME.git
```
This should take at most ~5 mins. After cloning the repo, you will be able to run the notebooks included in `demo`. Note that `software` and `demo` must remain in the same directory. Git LFS is required due to the large size of the network weight files; a standard Git clone may be used if pretrained networks are not required.


# Usage Instructions
1) Use `Prediction/Predict_GeneralNet.ipynb` for predictions using a general network, or `Prediction/Predict_ExpNets.ipynb` for predictions using experiment specific networks (note, experiment specific networks are trained specifically for the AnDi Challege dataset). 
2) Ensure `data_path` points to a folder containing data in AnDi format.
3) Run all cells in the chosen notebook. The output will be in a folder in `Prediction`.

Prediction time can be drastically reduced by lowering the range of GMM components tested (e.g. in the function `AnalyseEnsembleProperties`, set `n_components = np.arange(1, 3)`).

# Contact
Solomon Asghar - solomon.asghar.20@ucl.ac.uk

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/solomon-asghar-12b3a0215/
[SciPy-badge]: https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white
[SciPy-url]: https://scipy.org/
[Python-badge]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[NumPy-badge]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[TensorFlow-badge]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
[TensorFlow-url]: https://www.tensorflow.org/
[SKLearn-badge]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[SKLearn-url]: https://scikit-learn.org/stable/
