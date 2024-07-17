# U-AnD-ME
U-Net 3+ for Anomalous Diffusion analysis enhanced with Mixture Estimates. 

This repo contains code for the method that team "UCL SAM" used for the AnDi Challenge 2024.

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


# Usage instructions
1) Clone using Git LSF.
2) Use `Prediction/Predict_GeneralNet.ipynb` for predictions using a general network, use `Prediction/Predict_ExpNets.ipynb` for predictions using experiment specific networks (note, experiment specific networks are trained specifically for the AnDi Challege dataset).
3) Ensure `data_path` points to a folder containing data in AnDi format.
4) Run all cells in the chosen notebook. The output (in AnDi format) will be in a folder in `Prediction`.


If any issues arise, please contact me at solomon.asghar.20@ucl.ac.uk
