Contains modules for data loading, Model architecture, Training and Evaluation.

### Module Summary
- class_wise_results.py: Calculates the class-wise model accuracy.
- data_loader.py: Used to import CIFAR10 dataset and create a PyTorch data loader
- data_transform.py:  Data transformations for augmenting/pre-processing training and test datasets
- model_trainer.py: Main program that calls the training process, defined with Optimizers, # of Epochs, LR Schedulers
- network.py: Our VGG-based Neural Network architecture
- testing.py: To test the trained model
- training.py - Running the model training, caclulating the taks
- visualize_data.py: Visualize the Result of Training/Test datasets.
