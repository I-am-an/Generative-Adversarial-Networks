# Generative-Adversarial-Networks
##Abstract
In this project we aim to implement a GAN ( Generative Adversarial Network ) using different architectures and compare their performance to determine a more stable architecture for generating artificial images. The models are implemented in Python using Keras library. 
We had started with training the model to generate an image of handwritten numbers using mnist dataset available in kaggle and attained decent accuracy. The accuracy/loss graph were plotted to compare performances of different architectures.
The DCGAN (Deep Convolutional GAN) with a set of constraints on the architectural topology and certain loss functions makes the model more stable and accurate, details of  which are discussed in detail in this paper.

##Introduction
Deep Learning techniques have the potential to drastically change the world that we are a part of. There are still lots of areas for the application of deep learning to be explored.
One such area is the Artificial image generation where a specific type of neural network (ie) Generative Adversarial Network (GAN) is capable of generating artificial images that look closer to real ones. Such networks are highly useful for generating augmented data for training complex neural networks whose performance is limited by the number of training samples. The GANs can further identify patterns in images and can improve the resolution of images, in fact this is an active area of research where big companies like Facebook, Nvidia are scratching their brains. In this project we try to implement one such GAN for generating artificial images similar to that taken from a satellite. We experiment with different neural network architectures and state-of-art techniques to improve the performance of GANs.

##Literature Review
###1. Sample generation based on a supervised Wasserstein Generative Adversarial Network for high-resolution remote-sensing scene classification
By Wei Han, Lizhe Wang
They have used Wasserstein distance which is used to measure the difference between the generator distribution and the real data distribution. This addresses the problem of the gradient disappearing during sample generation, and distinctly promotes a generator distribution close to the real data distribution. An auxiliary classifier is added to the discriminator, guiding the generator to produce consistent and distinct images. With regard to the network structure, the discriminator and the generator are implemented by stacking residual blocks, which further stabilize the training process of the GAN-RSIGM. Extensive experiments were conducted to evaluate the proposed method with two public HRRS datasets. The experimental results demonstrated that the proposed method could achieve satisfactory performance for high-quality annotation sample generation, scene classification, and data augmentation.
Advantages of Wasserstein GAN
This can be used even if the generator distribution and the original distribution are disjoint. Gradients can always be observed. It calculates the minimum cost of transforming the generated noise distribution to the original probability distribution.
###2. Wasserstein GAN
By Martin Arjovsky , Soumith Chintala , and L´eon Bottou
Training WGANs does not require maintaining a careful balance in training of the discriminator and the generator, and does not require a careful design of the network architecture either.

Benefits of standard WGAN over Gtandard GAN
●	a meaningful loss metric that correlates with the generator's convergence and sample quality
●	improved stability of the optimization process

###3. Generative Adversarial Networks (GANs): An Overview of Theoretical Model, Evaluation Metrics, and Recent Developments
By Pegah Salehi, Abdolah Chalechale, Maryam Taghizadeh
●	WGAN method uses Earth-Mover (EM) or Wasserstein-1 distance estimation instead of JS divergence.
●	Compared to other distance metrics in distributed learning, Earth Mover (EM) distance produces stronger gradient behaviours. Lipschitz constraints approach provided a weight clipping method.
●	WGAN can produce unwanted results or not converge when using weight clipping in the discriminator. To apply the Lipschitz restriction, WGAN with Gradient Penalty (WGAN-GP) was therefore suggested.

##Dataset
Earlier the system used the open source MNIST dataset (available in Kaggle) to test the working of GAN. It consists of 60,000 samples of 28 X 28 binary images of handwritten digits with corresponding labels (0-9). However only the input images would be used training. This dataset will be replaced with satellite images in upcoming phases.
The system currently uses the open source Satellite Images of Water Bodies - dataset (available in Kaggle) to test the working of GAN. It consists of 2,481 samples of 656x657 RGB Images of Satellite Images of Water Bodies for Image Segmentation which contains RGB images and masks. However only the RGB input images would be used for training. The original 656 X 657 images are scaled down to 128 x 128 for implementation.


##Wasserstein GAN
	 After reading other papers ([1],[2]), we found that unlike other neural networks, The performance of GAN is dependent on how well the discriminator understands real and fake samples. The earlier used Sigmoid activation function in discriminator output is replaced with linear activation function to get a large scale of values of how discriminator understands the input image, this is used to compute Wasserstein Loss instead of cross entropy loss used earlier. The model is trained to minimise this Wasserstein loss by RMS propagation algorithm. The Wasserstein loss is found to give more insights for generator to generate artificial images and hence accuracy increases. Further the GAN with Wasserstein Loss (Wasserstein GAN/ WGAN) is found to be more stable while training.
	In order to increase stability further, we have implemented weight clipping technique (based on [3]) to force the weights of convolutional filter kernels to lie between -0.01 to 0.01. This further prevents overfitting and smooths out the resulting image to look like a real-world image. We are still working on improving the WGAN and our current results are shown below.
 


##References

1.	Sample generation based on a supervised Wasserstein Generative Adversarial Network for high-resolution remote-sensing scene classification: Wei Han a,b, Lizhe Wanga,b,⇑, Ruyi Feng a,b,⇑, Lang Gao a,b, Xiaodao Chen a,b, Ze Deng a,b, Jia Chen a,b, Peng Liu c
2.	Wasserstein GAN By Martin Arjovsky , Soumith Chintala , and L´eon Bottou: Courant Institute of Mathematical Sciences, Facebook AI Research
3.	Generative Adversarial Networks (GANs): An Overview of Theoretical Model, Evaluation Metrics, and Recent Developments: Pegah Salehi, Abdolah Chalechale, Maryam Taghizadeh


##Link to the Colab Notebook containing code

https://colab.research.google.com/drive/1zo6zQn3yhrPy537Kz3FTM0W4ui6pPs11?usp=sharing
