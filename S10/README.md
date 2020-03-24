# Session 10 Summary:
* Learning Rate(alpha):
	- Relationship between Loss and Wieght changes.
	- Not using Learning Rate means dangling of Loss function.
	- Constant LR causes plateau on Loss function(Simple SGD)
	- Hence we use the "mini-batch" Gradient Descent to reduce the Loss value. Hence it make sense to have a uniform random distribution of classes in a batch.
		E.g. for ImageNet dataset that has 1000 classes, we can't have a Batch size of 1000(computationally inefficient), but we can have FOUR consecutive batches of size 250, and then use SGD to calculate the MEAN of the losses of the Mini-Batch. Then, we use this Loss value to calculate the Weights of all the images in that Batch.
	- We use Adaptive LR, used by Adam, Adagrad(which also uses diff. alpha for diff. weights)

* Gradient PERTURBATION:
	- LR can cause Plateau. To avoid Plateau, we can use "Gradient PERTURBATION".
	- Adds the Gaussian Noise to the Gradient update equation, to jump out of the Plateau curve.
	- Can be implemented using PyTorch's torch.randn(tensor.size())*StdDev. + Mean.
	- We indrectly are doing the Noise introduction to Gradient function. How?

* Momentum and "Nesterov Momentum":
	- In previous approaches we were changing the Network by Calculating the Gradient ourselves and adding it to the function.
	- Using momentum, we utilize the past "few" gradient loss values to create a "sort of" momentum and use it for calculating the updated weights.
	- In MOMENTUM Algorithm, 
		* 1st STEP: Uses the correction factor called μ (Mu) to take the portion of previous weight(say 90%) and calculates the new weights.(μ can vary from 0.75 to 0.99)
			- Smaller the μ value, sooner the previous gradient value is discarded. Closer it is to 1, their effect continues for longer time.
			- μ is conidered to be a HyperParameter and should be tuned for every application.
		* 2nd STEP: Calculates the New Weight using the Correction factor calculated using STEP-1.

* Nesterov Momentum:
	- Uses the "LOOK AHEAD" gradient update mechanism.
	- It First calculates the Intermediate weight update using μ as in regular Momentum algorithm.
	- And then, again Calculate the Gradient of the intermediate weights to know the future weight value.
	- Hence we always have a look-up of the one step ahead for the weight.

* RMSProps: We look into the methods where we calculate the alpha(Learning Rate) differently for every parameters.
	- Since we need to calculate the momentum for every parameters of the network(could be 10 millions or so), it surely is not a Memory Efficient(GPU) technique.
	- In RMSProps, we calculate the "Relative Speed" of the gradient using the "Moving average of squared loss gradient", And then the "Actual" learning rate is adjusted for every "Single" parameter using the "Reciprocal of the Square root" of "Relative Speed" calculated previously.
		* Reference: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

* Adam: It's an adaptive algorithm which also focuses on the momentum change in the LOSS function along with weights(done by RMSProps)
	- Utilizes the concept of Momentum for LOSS(gradient) as well. Previous approaches focused on WEIGHT's momentum only.
	- Using the momentum of LOSS function itself, we'll figure out the effect of training on LOSS function(whether it's increasing or reducing), based on which the WEIGHT update equation is also changed.
	- It has a g(t) function for that

* Summary:
	- SGD uses fixed Learning Rate(alpha) for each Parameter
	- RMSProps uses different LR(alpha) for every single paramter, by using the momentum(variation is value) of the weights and figure out the current momentum based on it.(if it should move faster OR slower)
	- Adam works on top of RMSProps and calculates the momentum for LOSS function as well.
	- SGD Limitation: SGD can only be used for Object Detection and Object Recognition. It can't be used when,
		1. Less data samples for training
		2. Using Transfer Learning(rather Adam etc. is used)
		3. Reinforcement Learning(Adam)
		4. NLP, LSTM or RNN.
		5. GANs(Adam is preferred)
		6. Q-Learning with Function approximation

	- SGD Benefits:
		1. Less memory requirement as only 1st momentum is needed.
		2. Better Regularization

# Completely Different Approach: "Don't decay the Learning Rate, Increase the Batch Size!"
	* Both of them reduce the Loss function, but 2nd is memory intensive.
	* Training should start from Large Learning rate, as our randomly generated weights will be far from Optimal value.
	* We then reduce the learning rate during training to do smaller updates to our weights

# Cyclic Learning Rate(Beginning): Leslie N. Smith(Paper: Cyclical Learning Rates for Training Neural Networks - https://arxiv.org/abs/1506.01186)
	* Trick is to train the network(Only 1 epoch containing MULTIPLE Batches) starting from Low LR and increasing it exponentially.
	* At a point where the LOSS function reduces drastically, we note that LR value.
	* And then we can use that value of LR as our starting Learning Rate.
	* We can also use a slightly smaller learning rate that falls before the optimal learning rate. This is to WARM-UP our network.

	### LEARN: What is OneCyclePolicy?

# Tasks:
	* Write custom method to identify the Learning Rate "Plateau"
	* ReduceLROnPlateau(build-in for Keras)
	* Also for PyTorch :D
	* It's called using, "torch.optim.lr_scheduler.ReduceLROnPlateau"

Assignment: 

	* Pick your last code
	* Make sure  to Add CutOut to your code. It should come from your transformations (albumentations)
	* Use this repo: https://github.com/davidtvs/pytorch-lr-finder (Links to an external site.) 
	* Move LR Finder code to your modules
	* Implement LR Finder (for SGD, not for ADAM)
	* Implement ReduceLROnPlatea: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau (Links to an external site.)
	* Find best LR to train your model
	* Use SDG with Momentum
	* Train for 50 Epochs. 
	* Show Training and Test Accuracy curves
	* Target 88% Accuracy.
	* Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
	* Submit
