# neuralnet-from-scratch

## motivation
This project was meant to start understanding machine learning from a theoretical standpoint, rather than from a practitioner's one. It was implemented over four weeks, and achieves around 85% accuracy on classifying an author based on a text sample (a short story).

## theory

The neural network is simply a single layer (theta) for logistic regression. It calculates loss based off of cross entropy and softmax, and uses stochastic gradient descent to train.  

It works by taking the feature vector (bag of words + TF-IDF) and feeding it into theta (a big matrix) and outputs a vector. Softmax is then applied to this output, and the final vector can be interpreted as corresponding to how strongly the model believes each class is to be the actual label. Then, by simply taking the largest probability in this vector we have our actual prediction for the author.

To train the model, we evaluate it's performance by calculating the difference (error) between this output vector with the actual label (one hot encoded) using cross entropy. The final loss function is simply the sum of this error over all training samples added with the norm of theta, in order to keep the weights small and avoid overfitting. To minimize loss, we derive the gradient of this error function between the two distribuitions (the actual label and the predicted vector). After deriving the gradient of the error function the gradient of the final loss function becomes trivial, as simply summing these gradients over all samples together, with the gradient of the norm (simply the matrix itself). This gradient can then be subtracted from the original matrix in order to decrease loss, and by doing this iteratively the classification accuracy should get higher and higher. This process is known as gradient descent. 

## implementation 

We use numpy only, as this is "neural net from scratch". First, preprocessor.ipynb calculates the tf-idf of each sample by the  necessary components into a corresponding .txt file, and then outputting the resulting feature vectors into a binary numpy file, which effectively saves the feature vector into a file. Next, main.ipynb takes that vector and implements stochastic gradient descent, with the helper functions of the gradient of the loss and predicting a label. Finally, it uses some nice visualization and output helper functions to output results.



![results of experiments](https://i.imgur.com/W2ptpSa.png "Results of this project")
