# Brain Tumor Detection via Supervised Autoencoder Classifier

Brain tumors are caused by the growth of abnormal cells in the brain and my interfere with the normal brain function in vision, motor skills, and hormonal levels. Due to the challenges that brain tumors pose for the patients and the healthcare industry early detection of tumors for early intervention is of great importance. Due to the advances in medical data collection and machine learning algorithms, tumor detection by machine learning algorithms based on Convolutional Neural Networks (CNNs) has been investigated. 

In this project,a tumor detection classifier based on CNNs and the autoencoder architecture has been developed and compared with [Squeezenet](https://doi.org/10.48550/arXiv.1602.07360) an effective and time-efficient CNN-based classification architecture. 

---
### Dataset
In this project, the models are applied to MRI images of brain tumors, which can be found [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). The images are presented as a binary classification problem. an example of the images. A sample of the present images are presented below. 

![Alt text](https://storage.googleapis.com/kagglesdsdata/datasets/1608934/2645886/Testing/meningioma/Te-meTr_0000.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230109%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230109T204007Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=8b461b213e06c424686d2c296c57e84fac6e7d43335c6761501fd7e482141b5d9e873cc3b1ba3100542cbb7ddf44e484949d8aeebc5d54c8cd5d7ba26c4442eea8016ed7785263bafdb59b83793fdae83e07b78deab9910f80cb2a3a546b16c7f586efe18816363e1140d5022a17227bdec4bc3fdda454e656f517e9bd370830c8d9163890de101a2a8dc9802f96dde53a9684fb80d5011c9a2800fcc9ad9a90ea04f4f151273c2105a5a61d6af16f64af4f22a5799d89b4cd4e332b635a54e2cc877975db03fd2f137e59e8e9768e6f8195f4f57d65c96967c4331f0bf6a06c5e4a06e882ccd106ed5403b4850902a087d8a355e78fe00eb56a86e687f01a79 "a title") 
![Alt text](https://storage.googleapis.com/kagglesdsdata/datasets/1608934/2645886/Testing/meningioma/Te-meTr_0007.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230109%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230109T204007Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=34d4575d80edbcfb6519d9fdde894beb3f689a7833a571a5071754229f3e1c815284b537851772f1eac80f3d572d294c6e5d5f696df36449433511c24b1fa3565a3ba7109be5c376b7362c379a4757bfbc82d63249b9a481487f052f04fdbce19a7859fb558417e2f41674b18d6bdaaf13f1a4c25379bb6237b356b6ce01739717044c816335271317c9812be9ced8e5d15e9e3572a545828e0c1b0859fe86ddeccecf14e6cbc5d48e82668955eb26208491db486ffec8eb4d9cd7d43e9e2591d448d169cefb8aaf087ad63d2e32435ead7e0c02fcb2d90a43446dcb632a33e7a43337a3ea8140b6c1813ee346c14e3294f943af82ac2b4fb7f8370dd34e9877 "a title")

---
### Model 
A robust representation of the input data is a crucial step in developing classification algorithms. Autoencoders are neural networks that are associated with their ability in obtaining a robust representation of then data by reconstructing the input data through non-linear layers. In this project, convolutional autoenocders are the backbones of the suggested architecture for classifying the brain MRI images. 

Typically, the obtained representations by autoencoders are fed to a classification layer after they are obtained. However, based on the principals of multi-task learning, it is possilbe to integrate the classifications process to the unsupervised learning setting of autoencoders by adding another layer to the obtained latent representations. Therefore, this model, known as the supervised autoencoder can guide the representations towards better representing the data with respect to the ultimate goal, i.e., classification, which is utilized in this project.In addition, a resampling mechanism, in order to obtain a balanced distribution of data from each classes in the training process has also been applied. The general structure of the model is as follows:


          Layer (type)               Output Shape         Param #
    ================================================================
            Conv2d-1           [-1, 32, 30, 30]             896
            Conv2d-2           [-1, 64, 28, 28]          18,496
            Linear-3                  [-1, 256]      12,845,312
            Linear-4                   [-1, 32]           8,224
            Linear-5                  [-1, 256]           8,448
            Linear-6                [-1, 50176]      12,895,232
     ConvTranspose2d-7           [-1, 64, 30, 30]          36,928
     ConvTranspose2d-8            [-1, 3, 32, 32]           1,731
            Linear-9                    [-1, 2]              66
    ================================================================
    Total params: 25,815,333
    Trainable params: 25,815,333
    Non-trainable params: 0
    
---
### Results
The results of the convolutional supervised autoencoder has been compared with the pretrained and fine-tuned Squeezenet model. The highest performance in terms of accuracy for convolutional supervised autoencoder is 87%. However, the highest accuracy for Squeezenet only reaches to 57% under the same experimental protocols. This indicates the promising performance of utilized convolutional supervised autoencoder architecture in classifying MRI brain images in tumor detection. 

