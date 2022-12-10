# Deep-Learning-with-fashion_mnist-model-comparison
Machine learning is the study of how to program computers to learn and behave like humans without explicit programming. Training data is used to train machine learning algorithms. They are able to make precise predictions and choices based on historical data when fresh data is received. There are 4 types of machine learnings:

Supervised Learning
Supervised learning is a subset of machine learning and artificial intelligence is supervised learning, commonly referred to as supervised machine learning. It is distinguished by the way it trains computers to accurately classify data or predict outcomes using labeled datasets. The model modifies its weights as input data is fed into it until the model has been properly fitted, which takes place as part of the cross-validation process. A training set is used in supervised learning to instruct models to produce the desired results. This training dataset has both the right inputs and outputs, enabling the model to develop over time. The loss function serves as a gauge for the algorithm's correctness, and iterations are made until the error is sufficiently reduced. 	
Several supervised deep learning models were built including ResNet50, VGG16 and two models which are built from scratch. We used Transfer Learning for Resnet50 model and VGG16 model.
ResNet50
ResNet stands for Residual Network and it was introduced in 2015 by He Kaiming et al (2015)[ ]. Convolutional neural network ResNet-50 has 50 layers (48 convolutional layers, one MaxPool layer, and one average pool layer). Artificial neural networks (ANNs) of the residual kind build networks by stacking blocks of residual information.      

VGG-16
K. Simonyan et al (2014) [ ] put forth the convolutional neural network model known as VGG16 in their paper titled "Very Deep Convolutional Networks for Large-Scale Image Recognition." In ImageNet, a dataset of more than 14 million images divided into 1000 classes, the model achieves top-5 test accuracy of 92.7%. One of the well-known models submitted to ILSVRC-2014 was this one. It outperforms AlexNet by sequentially substituting multiple 33 kernel-sized filters for big kernel-sized filters (11 and 5 in the first and second convolutional layers, respectively).

CNN Model 1

This model has a total of 3 number of layers, it contains   4 convolutional layers,    3 maxpooling layers, 4 dense layer are included. test accuracy is 87.98%.


Unsupervised Learning

Unsupervised learning is when the model is trained with unlabeled data and learns from itself without supervision. Unsupervised learning, commonly referred to as unsupervised machine learning, analyzes and groups unlabeled datasets using machine learning algorithms. These algorithms identify hidden patterns or data clusters without the assistance of a human. It is the best option for exploratory data analysis, cross-selling tactics, consumer segmentation, and picture identification because of its capacity to find similarities and differences in information. 
