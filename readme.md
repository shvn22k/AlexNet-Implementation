
---

# **AlexNet Implementation from Research Paper**

## **Overview**
This repository contains a PyTorch implementation of the famous **AlexNet architecture**, based on the  [2012 research paper -  ImageNet Classification with Deep Convolutional Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3065386). The implementation was built from scratch without relying on pre-trained models, focusing on understanding the architecture and training pipeline as described in the paper.

This project demonstrates:
- The use of **convolutional layers, max-pooling, dropout, and ReLU activation**.
- **Custom data augmentation** techniques inspired by the paper, such as PCA-based color augmentation.
- Training on a custom dataset (insect classification).

---

## **Features**
- Full PyTorch implementation of AlexNet, following the exact and original architecture.
- Support for **binary and multi-class classification**.
- Custom **data augmentation pipeline** including resizing, cropping, flipping, and PCA augmentation.
- Example training on a custom insect dataset.

---
## **My Views and Findings**
1. The main implementation of Alexnet is in [```alexnet_implement-colab.ipynb```](https://github.com/shvn22k/AlexNet-Implementation/blob/main/alexnet_implement-colab.ipynb) . This one was trained on google colab's T4 GPU. Some slight changes I had to make different from the original paper include"
    - Changing the drop out values to ```0.7``` from ```0.5```, because for some reason with dropout set to ```0.5```, the model was not learning at all and giving stagnant losses throughout each epoch during training. Switching it to ```0.7``` showed significant gradual decrease in losses and improvement in accuracy from ```45.6%``` to ```79.69%```.
    - Ignored the softmax ending layer because pytorch's ```CrossEntropyLoss``` expects raw logits and handles ```softmax``` internally. 
    - Used a custom and smaller dataset to reduce training time and other complexities. Also, original imagenet dataset too large to handle. 
2. Main challenges I faced can be clearly seen in [```alexnet-implement.ipynb```](https://github.com/shvn22k/AlexNet-Implementation/blob/main/alexnet-implement.ipynb) and [```alexnet-implement-trialx```](https://github.com/shvn22k/AlexNet-Implementation/blob/main/alexnet_implement-trialx.ipynb) files since I tried training the network on my local GPU (rtx 3050).
    - For some unknown reason, no matter how much I change the dropout factor, learning rates and random seeds, the model was not learning. Even on running the same colab notebook locally, the results were poor.
    - Another challenge I faced was the issue of varying accuracies, each time I ran the accuracy block, I got different accuracies - ended up fixing this one by setting seed to 42 each time randomization was being performed. (shout out to claude for this one)

Feel free to download and run the code yourself and check if the issue persists in all local GPUs.

Big shout out to chatgpt and claude for helping me understand the paper in depth (annotated notes by me can be found in [```AlexNet Annotated - shiven.pdf```](https://github.com/shvn22k/AlexNet-Implementation/blob/main/AlexNet%20Annotated-%20shiven.pdf))

---

## **Acknowledgments**
- Original Paper: *ImageNet Classification with Deep Convolutional Neural Networks* by Alex Krizhevsky.
- Datasets: Custom insect classification dataset.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for details.

---
