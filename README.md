# Code for Poisson Image Interpolation (PII)

### Abstract:
Supervised learning of every possible pathology is unrealistic for many primary care applications like health screening. Image anomaly detection methods that learn normal appearance from only healthy data have shown promising results recently. We propose an alternative to image reconstruction-based and image embedding-based methods and propose a new self-supervised method to tackle pathological anomaly detection. Our approach originates in the foreign patch interpolation (FPI) strategy that has shown superior performance on brain MRI and abdominal CT data. We propose to use a better patch interpolation strategy, Poisson image interpolation (PII), which makes our method suitable for applications in challenging data regimes. PII outperforms state-of-the-art methods by a good margin when tested on surrogate tasks like identifying common lung anomalies in chest X-rays or hypo-plastic left heart syndrome in prenatal, fetal cardiac ultrasound images.

### Code:
Basic example provided in fpiSubmit.py  
Note that code was converted from tf1 to tf2. Some of the converted operations are not very efficient and will be redone shortly. Self-supervised task is written entirely in numpy for easy conversion to other frameworks.    

### Key files:  
self_sup_task_poisson.py - FPI/PII operation used to create self-supervised task
poissonBlend.py - Poisson operation to blend patches  
fpiSubmit.py - training/testing loops  
models/wide_residual_network.py - network architecture  
readData.py - data processing, reading/writing data 

### Data:
We use chest X-ray data available below from ChestX-ray14, provided by the National Institutes of Health:
```
https://nihcc.app.box.com/v/ChestXray-NIHCC
https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf
```
Specific patients used for training and testing are specified within train_lists and test_lists

### Author Information
```
Jeremy Tan, Benjamin Hou, Thomas Day, John Simpson, Daniel Rueckert, and Bernhard Kainz.: Detecting Outliers with Poisson Image Interpolation. (2021)

j.tan17@imperial.ac.uk
```


