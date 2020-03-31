Contains modules for data loading, Model architecture, Training and Evaluation.

### Module Summary
- class_wise_results.py: Calculates the class-wise model accuracy and Plots a comparison chart.
- data_loader.py: Used to import CIFAR10 dataset and create a PyTorch data loader.
- data_transform.py: Data transformations for augmenting/pre-processing training and test datasets.
- model_trainer.py: Main program that calls the training/validation process, fits the data into the Model, defined with Optimizers, # of Epochs, LR Schedulers.
- resnet.py: Imported the RESNET18 model architecture from reference
- testing.py: To test the trained model and plot a training/testing graph for accuracy and losses per epoch.
- training.py - Running the model training, caclulating the model accuracy and losses.
- utils.py - progress_bar module to show the training and valiation progress.
- visualize_data.py: Visualize the Result of Training/Test datasets.
- grad_cam_vis.py: takes the model and 'image_list' as array of image dict objects. {image, target_label, prediction_label}
- lr_utility.py: lr_finder utility for optimum Learning Rate
- misclassifieds.py: Takes model and testloader with classes and plots the mis-classified samples
