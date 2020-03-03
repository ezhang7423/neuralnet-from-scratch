# neuralnet-from-scratch

## motivation
This project was meant to start understanding machine learning from a theoretical standpoint, rather than from a practitioner's one. It was implemented over four weeks, and achieves around 85% accuracy on classifying an author based on a text sample(a short story).

## theory

The neural network is simply a single layer (theta) for logistic regression. It calculates loss based off of cross entropy and softmax, and uses stochastic gradient descent to train. Here's how it works: the feature vector(bag of words + TF-IDF) is fed into theta (a big matrix) and outputs a vector. Softmax is then applied to this output, and the final vector can be interpreted as corresponding to how strongly the model believes each class is to be the actual label. By taking the largest probability in this vector we have our actual prediction for the author. To train the model, we evaluate it's performance by calculating the difference(error) between this output vector with the actual label using cross entropy. Next, we derive the gradient of this error function in order to minimize the difference between the two distribuitions(the actual one hot encoded vector and the predicted vector). The final loss function is simply the sum of this error over all training samples added with the norm of theta, in order to keep the weights small and avoid overfitting. After deriving the gradient of the error function the gradient of the final loss function can be found by summing these gradients over some samples together. This gradient can simply be subtracted from the original matrix in order to decrease loss, and by doing this iteratively the classification accuracy should get higher and higher. This process is known as gradient descent. 

## implementation 

First, preprocessor.ipynb calculates the tf-idf of each sample by the  necessary components into a corresponding .txt file, and then outputting the resulting feature vectors into a binary numpy file, which effectively saves the feature vector into a file. Next, main.ipynb takes that vector and implements stochastic gradient descent. It does this with the helper functions of the gradient of the loss and predicting a label. Finally, it uses some nice visualization and output helper functions to output results.


with the cross entropy as the loss function. A
![results of experiments](https://i.imgur.com/W2ptpSa.png "Results of this project")
