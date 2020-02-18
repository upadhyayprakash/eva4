Python Notebook Link: https://colab.research.google.com/drive/10bihZ-sdE1UK5NzMvOzx-x7w2CfyqJX2#scrollTo=p2XVdfv0M6Ll

## Targets:
	- To Reduce the Model Parameters
  - Increase it's Robustness towards end of the Architecture
  - Also make it rotation Invariant by using Image Augmentation(RandomRotation of -15, +15 degrees)
  - Simplify the Overall Model Architecture using Sequential Modeling
  
## Results:
  Train Accuracy: 98.43%
	Test Accuracy: 99.44%
	Total Parameters: 9880
  
  - Number of Parameters were reduced using fewer channels for a small sized dataset.
  - Another layers of Convolution was introduced to make sure the model is learning till the end(i.e. after the GAP layer)
  - Rotation invariance were inreased to +-15 degrees, looking at few of the images that were tilted more than others.

## Analysis:
  - Dropout is used to make sure the model converages smoothly towards the end of the epochs.
  - Reducing the parameters made sure that model trains faster along with high Batch_Size
  - Image Augmentation worked as expected, the close-up view of the Train/Test images were useful.

## File Link: https://github.com/upadhyayprakash/eva4/blob/master/S5/EVA4_Session_5_MNIST.ipynb
