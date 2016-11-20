# ANN (Theo Satloff and [@WalkerGriggs](https://github.com/WalkerGriggs))

# Abstract:

NB: Thanks to our friend CP ([@luftdanmark](https://github.com/luftdanmark)), who was kind enough to provide us the tic-tac-toe dataset used herein.

This dataset represents the full set of possible, terminal board configurations for the end of tic-tac-toe games, where player #1 is assumed to have played first. Each column of the dataset represents one single square on a tic-tac-toe game board. Each value within said columns represents which player occupied that space during the game. The players are represented by either 1 or 2, and if no player ever enters a specific square, a 0 is placed therein. It is important to note that only games that resulted in a win for player 1 or player 2 are included; there are no game states that resulted in a tie game.

We split this dataset into a training set and a test set by using ¾ of the data to train, and the remaining ¼ to test. Our goal is to create a neural network that can learn the patterns of the tic-tac-toe board to guess the outcome accurately more than 80% of the time. While in practice, 80% is not a particularly striking number, we feel that due to the kind of dataset and semi randomness involved, 80% proves to be a good classification. We planned to run the dataset through an auto-encoding, neural network that is built to use forward feed propagation and back propagation.

Using this data, we decided to implement a simple, 3 layer, artificial neural network. We tested to see if it could be efficiently trained to perform simple, yet meaningful tasks. We used each possible board combination of tic-tac-toe for our dataset, paired with a binary win/loss column. 75% of this data set was used to train our ANN, while the renaming 25% served as our test set.

Below are graphs for each variable that is varied, while the others are held constant. Noticeable is the significant decrease in error as all the learning rate, momentum, and decay rate are decreased, but what is not apparent is the increased time the network took to train.

# Visualizations:

![alt tag](https://github.com/satloff/ann/blob/master/resultImages/HiddenNodes.png)
![alt tag](https://github.com/satloff/ann/blob/master/resultImages/LearningRate.png)
![alt tag](https://github.com/satloff/ann/blob/master/resultImages/DecayRate.png)
![alt tag](https://github.com/satloff/ann/blob/master/resultImages/Momentum.png)
![alt tag](https://github.com/satloff/ann/blob/master/resultImages/OnlineStochastic.png)

# Results:

![alt tag](https://github.com/satloff/ann/blob/master/resultImages/SampleResults.png)

Here are some of the mid-range results we saw. We were happy with our output of 211/250 (or ~84%) correct. We must note that the training set was quite small. On a larger set, the tests would have been more accurate, but there are indeed no more terminal configurations. But, for the same of speed, we chose a data set with only ~950 consumable patterns.

As can be easily seen in the output above, the far left number is the target value with an arrow pointing towards the predicted value (what our network has concluded), followed by if it guesses right/wrong. The last number shows the network guess rounded to a binary win/loss. We made the executive decision that a tic-tac-toe game doesn't need to be weighted in any way to accommodate false positives or negatives. Thus, anything greater than .5 was a win, and anything less is a loss.

By design, our neural network is a very flexible and adaptable network. In order to accomplish this specific task, we designed the network to take in a number of variables: the number of inputs, the number of hidden nodes, the number of outputs, the learning rate, the momentum, the decay rate, the activation type, and the gradient descent type. By having all of these controls be variable, the user is easily able to tune the network framework to work with most datasets.
Analysis:

As more nodes are added to the hidden layer, more complexity is added to the functions being fitted. While this principle is always true, we quickly learned that higher dimensionality is not always beneficial to the output. More complexity means more computation is required to fit. In addition, the more data being trained, the higher the likelihood that the data will be overfit.

The activation type (sometimes called a “squashing function”) was one of the most interesting pieces of this project in that this is the variable that allows for data to actually be trained efficiently. We decided to implement four activation functions: sigmoid, hyperbolic tangent, Elliot, and softmax. The sigmoid function allows for a progression of data over a curve that filters from a small beginning to a climax over time. Any cumulative distribution function is technically sigmoidal. The second activation function we chose was the hyperbolic tangent (tanh). Tanh evaluates the exact values of any given rational number when its argument is e. The last main function we implemented was the softmax, which allows for us to convert raw scores to probabilities. Softmax is like a logistic function that generalizes to multiple classes.

These activation functions most intriguing, but they are also risk confusion when deciding how they should interact with each other. For example, does sigmoid work best on the output layer or the hidden layer? Should tanh be used in sequence with sigmoid? How close is the Elliot approximation, and is it worth it? In the future, we would like to implement rectified linear units ReLUs, although on a different data set.

In a paper written by David Elliot, he discussed the efficiency of modified back-propagation and optimization methods. For our usage, Elliot supplied an approximation formula for a sigmoid that uses no exponent. This reduces the time spent on each iteration, speeding up training.

Our network uses a simple auto-encoding scheme. In other words, our network makes predictions using both forward and back propagation.

The next variable we had to decide upon was the type of gradient descent. We chose to implement both online and stochastic gradient descent. In the future, we would like to implement mini-batch gradient descent, as we believe this will reduce processing time and reduce our error. We would also like to decay our learning rate over time, but in practice we felt this was not necessary for this version of the project.

# Conclusion:

In retrospect, we should have chosen a dataset that was more useful in showing the power of the neural network.  To some degree, determining who was the winner of a tic-tac-toe game after the completion of the game is insufficiently useful, or effective in demonstrating the value of a trained neural network.  A far better example would have been to use a dataset that shows the outcome given the first square selected. In other words, based on a given square out of the possible 9, which is the one that is the best choice.  We could have then proceeded in each case to determine the optimal play.  Further, our dataset was suboptimal, as ties were excluded.

Despite the caveat listed above, this was a rewarding and educational exercise, and has us identifying datasets to train and test on a recreational basis.  We are also looking for entrepreneurial opportunities using what we have learned.

# Sources:

http://www.webpages.ttu.edu/dleverin/neural_network/neural_networks.html

https://www.willamette.edu/~gorr/classes/cs449/momrate.html

http://cs.colby.edu/courses/S16/cs251/LectureNotes/Lecture_33_NeuralNetworks_04_25_16.pdf

http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

http://www.amazon.com/Programming-Collective-Intelligence-Building-Applications/dp/0596529325

http://ufldl.stanford.edu/tutorial/

http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

http://neuralnetworksanddeeplearning.com/chap3.html

https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Activation_Functions
