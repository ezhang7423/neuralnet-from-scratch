# neuralnet-from-scratch

## motivation
This project was meant to start understanding machine learning from a theoretical standpoint, rather than from a practitioner's one. It was implemented over four weeks, and achieves around 85% accuracy on classifying an author based on a text sample (a short story).

## theory

The neural network is simply a single layer (theta) for logistic regression. It calculates loss based off of cross entropy and softmax, and uses stochastic gradient descent to train.  

It works by taking the feature vector (bag of words + TF-IDF) and feeding it into theta (a big matrix) which outputs a vector. Softmax is then applied to this output, and the final vector can be interpreted as corresponding to how strongly the model believes each class is to be the actual label. Then, by simply taking the largest probability in this vector we have our actual prediction for the author.

To train the model, we evaluate its performance by calculating the difference (error) between this output vector with the actual label (one hot encoded) using cross entropy. The final loss function is simply the sum of this error over all training samples added with the norm of theta, in order to keep the weights small and avoid overfitting. To minimize loss, we derive the gradient of this error function between the two distribuitions (the actual label and the predicted vector), and call this update. After deriving update, the gradient of the final loss function is simply the sum of updates over all samples, added to the gradient of the norm( which is just the matrix itself). This summed update can then be subtracted from the original matrix in order to decrease loss, and by doing this iteratively the classification accuracy should get higher and higher. This process is known as gradient descent. 

## implementation 

We use numpy only, as this is "neural net from scratch". First, preprocessor.ipynb calculates the tf-idf of each sample by calculating the  necessary components and saving in a corresponding .txt file. It then outputs the resulting feature vectors (a matrix at this point) into a binary numpy file. Next, main.ipynb takes that vector and implements stochastic gradient descent, with the helper functions of the gradient of the loss and predicting a label. Finally, it uses some nice visualization and output helper functions to output results.



![results of experiments](https://i.imgur.com/W2ptpSa.png "Results of this project")
