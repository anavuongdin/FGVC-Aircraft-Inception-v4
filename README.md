# FGVC-Aircraft-Inception-v4
- This project is deployed to train an Inception-based model on FGVC Aircraft dataset
- Work credited to: https://github.com/XuyangSHEN/Non-binary-deep-transfer-learning-for-image-classification
- This project has some modifications:
  - Add L2SP-Fisher to loss function so that model can carry even more knowledge from pre-trained Inception-v4
  - Add more learning rate schedulers for more experiments
- This project has a demo for both training and inferencing at experiments\drive\FGVC_Aircraft_AI.ipynb
- Trained weights of the model can be access from the following link: https://drive.google.com/file/d/1stnla0VTD4O8YERmWUqidqnNJF6NvhYF/view?usp=sharing