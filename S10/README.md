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
	- 
