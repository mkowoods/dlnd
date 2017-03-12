Notes From Andrew NG - Nuts and Bolts of Applying DL

Training Error High?(Bias, e.g. human-level is 1% and Train error is 5%)
 - Yes(try below and re-train):
  - Bigger Model
  - Train Longer
  - New Model Architecture
 - No
  - Dev Error High?(Variance, e.g. HL is 1%, Train Error is 2% and Dev Error is 6%)
   - Yes(try below and retrain):
     - More Data
     - Regularization
     - New Model Arch
   - No:
     - Done
