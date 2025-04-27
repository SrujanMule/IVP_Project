# IVP_Project
Project on GalaxyZoo data done during the IVP course at IISER P.

In astrophysics and astronomy, galaxy morphology is important to understand the formation and evolution of galaxies. But the very classification, even by humans, is not exact and is open to interpretability. Given the increasing availability of enormous amount of data, it becomes necessary to automate the process of classifying them. In this project, I would like to use CNN based architecture that mimics how humans classify the galaxies (i.e. the probability distributions, instead of the assuming that the highest probability is the true label). The input would be a coloured .jpg image of a galaxy and the output the probability distribution over the 37 questions that form the leaves of a decesion tree. One can, by assuming that the highest probability class is the true class during the testing, also predict the morphology type (The output of the first 3 class).
 
The data is obtained from the Galaxy Zoo 2 dataset in Kaggle that has 61578 datapoints/images for training. (https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data). We will split the data in the following way:
• Train: 80%
• Validation: 10%
• Test: 10%
 
Data augmentation (like horizontal flip, rotation, and clip) was planned to be used for better training (which would also increase the datapoints). But given the compute constraint, it hasnt been done.

 
My model is based on the architecture of EfficientNet. The architecture is relatively light compared to previous models like ResNets, VGG etc. and is computationally more efficient than others. It has been shown to have excellent performance on image classification tasks (on imagenet data) using a compound scaling approach that balances network depth, width and resolution. It takes in a 128 * 128 input of the image after rescaling it from the original dimension of 424*424 and outputs a 37 dimension vector that correspond to the probability distribution.

Make sure you have glob, os, torch, pandas, numpy, PIL libraris installed before running the code.

Some remarks:

1.  The train.py file uploaded is a generic file that is coded as per the instructions given. The original training of the model was done using 2 more different      loss functions (apart from the KL-Divergence over the first 3 class): MSE Loss and Focal Loss
      • MSE Loss since the eveluation metric of the original challenge was RMSE
      • Focal Loss over the first 3 class of questions since the model was expected to do well at the very least on the topmoost nodes (Class 1.1, 1.2, 1.3) to           classify the galaxies broadly into the 3 cateogories. This loss helps the model pay more attention to the examples that are difficult to learn, and take          care of any class imbalance in the data. (The 3rd class had extremely low number of datapoints)
      • The KL-Divergence was used only over the first 3 classes as well.
    
  A weighted average of these functions was used the final loss function. The weights were dynamically alloted based on the valeus of each loss so equal            attention was paid to reduce all of them.

2. I have employed Group Normalization technique inspired from the winning solution of the challenge (https://sander.ai/2014/04/05/galaxy-zoo.html), where the       probabilities of each class of questions are noramlized to sum to 1. This is obatined after applying ReLU over outputs of the network, during normalization.

3. The data folder also has images stored in the respective class and the probabilities of each example set of the class extracted in a seperate csv file.
   
4. Although the instructions were to provide 10 examples each of the number of classes we are trying to classify, it doesn't fit well given the rpobabilistic        nature of the task at hand. But since we expect the model to atleast do well in the first set of question in the decision tree (Class 1.1, 1.2, 1.3), I have      uploaded 10 images each of the first 3 class based on the argmax() over them. When one evaluates the moel based on this metric only over the first 3 classes,     the model had an accuracy of about 85% and RMSE ~ 0.23. The original bar of RMSE before the challenge was about 0.28 so this model is doing better than the       older model. But it comes no where close to the winning solution that had about ~ 0.07. This is probably beacuse I ahvent employed as many data augumentation     techniquies where all the outputs over the auguments were concatenated to provide a very rich representation. This wasn't feasible due to hardware constraint.
   
5. The model was trained over 20 epochs. This took about 8 hours on the local device. It can be trained for longer to get better results. (The validation error      hadn't plateued yet). AdamW optimizer with deacying learning rate was used as the optimiser.   


