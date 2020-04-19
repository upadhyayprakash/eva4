## **Session-12 Assignment**

### **Assignment A**:
Download this TINY IMAGENET (Links to an external site.) dataset. 
Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy. 
Submit Results. Of course, you are using your own package for everything. You can look at this (Links to an external site.) for reference. 

Target Validation Accuracy: 50%. 
### **Assignment B**:
- Download 50 images of dogs. 
- Use this (Links to an external site.) to annotate bounding boxes around the - dogs.
- Download JSON file. 
- Describe the contents of this JSON file in FULL details (you don't need to - describe all 10 instances, anyone would work). 
- Refer to this tutorial (Links to an external site.). Find out the best total - numbers of clusters. Upload link to your Colab File uploaded to GitHub. 

### **These are the questions in S12-Assignment-Solution**
- What is your final accuracy?
- Share the Github link to your ResNet-Tiny-ImageNet code. All the logs must be visible. 

- Describe the contents of the JSON file in detail. You need to explain each element in detail. 
- Share the link to your Github file where you have calculated the best K clusters for your 50 dog dataset. 

- Share the link to your 50 Dog Images Folder on GitHub
- Share the link to your JSON file on GitHub

#### **RESULT**
GitHub Source: https://github.com/upadhyayprakash/eva4/tree/master/S12

#### **Summary**

| Metric          | Values         |
| --------------- | -------------- |
| Train/Test Split| **70/30 Split**  |
| Network Arch.   | **RESNET-18**  |
| # of Epochs     | **50**         |
| # of Parameters | **11,173,962** |
| Final Accuracy  | **__.__%**    |
| Optimizer | **SGD** with Momentum |
| Scheduler | **OneCyclePolicy** |
| K-MEANS Method | **Elbow** Method |
