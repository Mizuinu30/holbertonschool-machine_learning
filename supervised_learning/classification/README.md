## Classiification using Neural Networks
Learning Objectives

    What is a model?
        A machine learning model is a program that can find patterns or make a decision from a previously unseen dataset.
    
    What is supervised learning?
        Is a category of machine learning that uses labeled datasets to tarin algorithms to predict outcomes and recognize patterns.
    
    What is a prediction?
        Is the output of an algorithm after it has been trained on a historical dataset and applied to new data when forecasting the likelihood of a particular outcome.

    What is a node?
        Is a place where computtion happens, loosely patterned on a neuron in the human brain, which fires when it encounters sufficient stimuli.

    What is a weight?
        A Weight refers to connection managements between two basic units within a neural network. to train these units to move forward in the network, weights of unit signals must be increased or decreased.

    What is a bias?
        Bias is a measure of how easy it is to get the perceptron(neuron) to output 1, Or to put it in more biological terms, the bias is a measure of how easy it is to get the neuron to fire.

    What are activation functions?
        An activation function determines the range of values of activation of an artificial neuron, this is applied to the sum of the weighted input data of the neuron. the Function it's characterized by the property of non-linearity.
        
        Sigmoid? - This Function takes any real value as input and outputs in the range of 0 to 1.

        Tanh? - This function outputs values in the range of -1 to +1, It's zero-centered.

        Relu? - The rectified linear unit or rectifier activation function introduces the property of nonlinearity to a deep learning model and solves the vanishin gradients issue.

        Softmax? - This function trasnforms the raw outputs of the neural network into a vector of probabilities, essentially a probability distribution over ht einput classes.

    What is a layer?
        Is a structure or network topology in the model's architecture, which takes information from the previous layers and then passes it to the nect layer.

    What is a hidden layer?
        Hidden layer or layers are the intermediary stages between input and output in a neural network, they are responsible for learning the intricate structures in data and making neural networks a powerful tool.

    What is Logistic Regression?
        is a supervised learning algorithm taht accomplisshes binary classification tasks by predicitng the probability of an outcome, event or observation.

    What is a loss function?
        Loss function or error function is a crucial component in ML that quantifies the difference between the predicted outputs of a ML algorithm and the actual target values.

    What is a cost function?
        The cost function is the technique of evaluating performance of our algorith/model. It takes both predicted outputs by the model and actual outputs and calculates how much wrong the model was in it's prediction.

    What is forward propagation?
        It Reffers to the calculation and storage of intermediate variables(including outputs) for a neural network in order from the input layer to the output layer.

    What is Gradient Descent?
        Is an optimization algorithm that's used when training a machine learning model. It's based on a convex function and tweaks its parameters iteratively to minimize a given function to its local minimum.

    What is back propagation?
        is the process of adjusting the weights of a neural network by analyzing the error rate from the previous iteration. Hinted at by its name, backpropagation involves working backward from outputs to inputs to figure out how to reduce the number of errors and make a neural network more reliable.

    What is a Computation Graph?
        Is a direct graph where the nodes represent mathematical operations and the edges represent the flow of data betwwn the operations.
        
    How to initialize weights/biases?
        Weight initialization serves as a cornerstone in machine learning algorithms, influencing model performance and convergence across diverse datasets and scenarios. Various initialization techniques, such as He-et-al and Kaiming, offer tailored approaches to initializing network parameters, catering to different activation functions and architectures, bias will depend only on the linear activation of that layer, but not depend on the gradients of the deeper layers. Thus, there is not a problem of diminishing or explosion of gradients for the bias terms. So, Biases can be safely initialized to 0.
    
    The importance of vectorization
        It offers several advantages, including improving the efficiency of text based ML models, enabling semantic analysis, and enhancing the understanding of textual data.

    How to split up your data?
         In a machine learning model is typically taken and split into three or four sets. The three sets commonly used are the training set, the dev set and the testing set. Data should be split so that data sets can have a high amount of training data. For example, data might be split at an 80-20 or a 70-30 ratio of training vs. testing data. The exact ratio depends on the data, but a 70-20-10 ratio for training, dev and test splits is optimal for small data sets.

    What is multiclass classification?
        A classification task with more than two classes, e.g., classifying a set of fruit images that may be oranges, apples or pears. Multiclass classification makes the assumption that each sample is assigned to one and only one label.

    What is a one-hot vector?
        With one-hot, we convert each categorical value into a new categorical column and assign a binary value of 1 or 0 to those columns. Each integer value is represented as a binary vector. All the values are zero, and the index is marked with a 1.

    How to encode/decode one-hot vectors?
        One-hot encode the labels into probability vectors by using the onehotencode function. Encode the labels into the first dimension, so that each row corresponds to a class and each column corresponds to a probability vector. Decode the probability vectors by using the onehotdecode function.

    What is the softmax function and when do you use it?
        The softmax function is often used as the last activation function of a neural network to normalize the output of a network to a probability distribution over predicted output classes.

    What is cross-entropy loss?
        Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label.

    What is pickling in Python?
        Pickling, in the context of machine learning with Python, refers to the process of serializing and deserializing Python objects to and from a byte stream. It allows us to store the state of an object in a file or transfer it over a network, and then restore the object's state at a later time.
