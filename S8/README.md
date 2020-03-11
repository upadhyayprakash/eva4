### **Session-8 Assignment**

1. Go through this repository: https://github.com/kuangliu/pytorch-cifar (Links to an external site.)
2. Extract the ResNet18 model from this repository and add it to your API/repo. 
3. Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
4. Your Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 
5. Once done finish S8-Assignment-Solution. 
 

#### **Tasks for S8-Assignment**

1. Share the link to your GitHub S8 code. Please make sure it is public. If your code is not modular or structured into different functional files, you will get 0 for the whole submission.
2. What is the final accuracy of your model
3. Paste your training or epoch logs here

#### **RESULT**
GitHub Source: https://github.com/upadhyayprakash/eva4/tree/master/S8
- Basic Augmentation Used: Random Horizontal Flip, RandomErasing


#### **Summary**

| Metric                | Values         |
| --------------------- | -------------- |
| *Network Arch.*   | **RESNET-18**  |
| *# of Epochs*     | **20**         |
| *# of Parameters* | **11,173,962** |
| *Final Accuracy*  | **89.290%**    |

#### **Code Structure**
- **modules:** contains the python modules used during model training, testing and visualizations
- **images:** contains charts/graphs image output of the model training/validation
