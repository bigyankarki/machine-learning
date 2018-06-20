<h3>Dataset:</h3> I made my own dataset that consisted images of four of my friends (i.e 4 classes). Each class had about 100 instances. Similarly, the images were collected from google photos. Sadly, due to its large size, the dataset could no be included in the repo. However, for curious cats, I would be happy to forward it. ;) Just shoot me an email/message!

<h3>Model:</h3> Pre-trained Inception V3 model was used to train the Convolution Neural Network. The fully connected layer was replaced with output layer of only 4 neurons.

<h3>Pre-processing:</h3> Data augmentation technique was used to increase the number of instances while training.

<h3>Training:</h3> This model was trained on an AWS EC2 g3.4xlarge Deep Learning Ubuntu 10.0 AMI instance.
<h3>Improvements:</h3> The accuracy of the model is currently about 67%, which is pretty bad. However, it could be increased using some of the tricks listed below:

<ul>
<li> Increasing training examples (the most effective and costly part of any ML problem. Bummer!)</li>
<li> Use regularization technique such as drop out </li>
<li> Current number of epoch is 10. For more accuracy, it may be increased to about 50 using Early Stopping technique.</li>
<li> Current Data Augmentation technique only uses rotations, zooms, and flips. We can augment our data by changing hue and brightness.</li>
</ul>
