# CNN Active Learning

This code allows to study active learning methods on several convolutional neural networks, with diferent image datasets, and other parameters, at once.

At the end of the experiment it shows the results in one or several plots depending on user arguments, and save them.

Therefore you can play with the results after, without relaunching all the experiment, that can be pretty long.

**Active learning methods**: least confidence sampling, margin sampling, and entropy sampling.

**Convolutional Neural Networks**: VggNet, ResNet and AlexNet.

**Image datasets**: CIFAR10 and CIFAR100.

## How to use this code?

Open a terminal, and:
- `git clone https://github.com/rafutek/CNN-active-learning.git`
- `cd CNN-active-learning`
- `pip install -r requirements.txt`
- `cd code`
- `python experiment.py -h` to display the help

## Examples

- Default experiment

  Launch the experiment with default parameters:
  ``` python
  python experiment.py
  ```
  Default main parameters are:
    - models: *VggNet*
    - datasets: *CIFAR10*
    - methods: *least confidence sampling*
    - Ks: *200*
    - number of trainings: *5*
    - number of epochs: *10*
   
   Therefore, you will have only one result.
  
  - Experiment several active learning methods

  Launch the experiment with margin and entropy sampling:
  ``` python
  python experiment.py --methods "margin,entropy"
  ```
  Main parameters are now:
    - models: *VggNet*
    - datasets: *CIFAR10*
    - methods: *margin sampling, entropy sampling*
    - Ks: *200*
    - number of trainings: *5*
    - number of epochs: *10*
   
   This will launch 2 active learnings, one per method, so the final plot will contain 2 results.
   
  - Experiment several active learning methods and models

  Launch the experiment with margin and entropy sampling on AlexNet and ResNet:
  ``` python
  python experiment.py --methods "margin,entropy" --models "AlexNet,ResNet"
  ```
  Main parameters are now:
    - models: *AlexNet, ResNet*
    - datasets: *CIFAR10*
    - methods: *margin sampling, entropy sampling*
    - Ks: *200*
    - number of trainings: *5*
    - number of epochs: *10*
   
   This will launch 4 active learnings, one per method and model.
   
  - Experiment several active learning methods, models, and numbers of samples to add to next training set

  Launch the experiment with margin and entropy sampling on AlexNet and ResNet, with k=100 and k=30:
  ``` python
  python experiment.py --methods "margin,entropy" --models "AlexNet,ResNet" --Ks "100,30"
  ```
  Main parameters are now:
    - models: *AlexNet, ResNet*
    - datasets: *CIFAR10*
    - methods: *margin sampling, entropy sampling*
    - Ks: *100, 30*
    - number of trainings: *5*
    - number of epochs: *10*
   
   This will launch 8 active learnings, one per method and model and k.
