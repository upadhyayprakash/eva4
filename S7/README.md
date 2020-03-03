### **Session-7 Assignment**
I. Make the [code](https://colab.research.google.com/drive/1qlewMtxcAJT6fIJdmMh8pSf2e-dh51Rw) Modular. Points to ponder:
1. Use GPU - Data and Model porting to GPU during training
2. Separation of Concern - load modules from different files
3. Easy to understand Program flow - Call network artifacts in main program
4. Readability with Comments(wherever necessary) - Explain the purpose of each code cell

II. Optimization:
1. change the code such that it uses GPU
2. change the architecture to C1C2C3C40 (basically 3 MPs)
3. total RF(Receptive Fields) must be more than 44
4. one of the layers must use Depthwise Separable Convolution
5. one of the layers must use Dilated Convolution
6. Use GAP (compulsory):- add FC after GAP to target # of classes (optional)
achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M. 
7. upload to Github


#### **RESULT**

Notebook Link: [Notebook](https://github.com/upadhyayprakash/eva4/blob/master/S7/EVA4_Session_7_CIFAR10_Modular.ipynb)
