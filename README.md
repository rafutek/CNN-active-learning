# CNN Active Learning

This code allows to study active learning methods on several convolutional neural networks, with diferent image datasets, and other parameters, at once.

At the end of the experiment it shows the results in one or several plots depending on user arguments, and save them.

Therefore you can play with the results after, without relaunching all the experiment, that can be pretty long.

**Active learning methods**: least confidence sampling, margin sampling, and entropy sampling.

**Convolutional Neural Networks**: VggNet, ResNet and AlexNet.

**Image datasets**: CIFAR10 and CIFAR100.

***
### How can to use this code?

Open a terminal, and:
- `git clone https://github.com/rafutek/CNN-active-learning.git`
- `cd CNN-active-learning`
- `pip install -r requirements.txt`
- `cd code`
- `python experiment.py -h` to display the help
