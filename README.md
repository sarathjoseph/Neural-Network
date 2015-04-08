# Neural-Network

A raw Neural implementation in Python that has a simple interface

<b>Usage</b><br/>

Instantiation <br/>

Neural_Network(num_inputs, num_outputs)  # create a neural network object passing in number of input and output nodes
<br/>
eg nn= Neural_Network(4,3) 
<br/>

Training <br/>

nn.train(train_data, num_of_epochs,learning_rate) 

<br/>
Train data is passed in as a 2d array of inputs vectors with the last entry being expected output. 
<br/>eg.  train_data= [[1, 0, 0],[0, 0, 1],[1, 1, 1]] where [1, 1, 1] is the expected output.

It returns the training error which can be useful for knowing how the network is learning
<br/>

Testing<br/>

nn.test(test_data) 

<br/>
Pass in test_data in the form of a 2d array with test input data . 
<br/>
eg
nn.test([0 , 1 , 0])

<br/>


Get weights - helps weights from the network
<br/>

  eg nn.get_weights()
  
  <br/>
  
  Set weights- helps set weights 
  <br/>
  Usage of weights should follow the same format (dictionary) that you obtain from doing a get_weights() on the network
  nn.update_weights(weights)
  <br/><br/>
<b>LANGUAGE IDENTIFICATION PROJECT<b><br/>

A Neural Network for Language Identification (Eng, Dutch , Italian ) using Python
Wikipedia Language Detection Feed­forward neural network with std backpropagation implementation

<br/>

<b>Classes</b>
<br/>
<ul>
<li>Neural_Network</li>
<li>Network_Trainer</li>
<li>Language_Features</li>

</ul>
<br/>
Language_Features class is instantiated with the feature set and contains the vectorize method which turns the features into real numbers. The feature set is passed in as callbacks for allowing addition of features dynamically to the Language Features class.

<br/>
Network_Trainer class is instantiated with the language features object , the codes and the input folder for training. The codes are kept in a dictionary containing the mapping between folder names (language names) and output codes
eg { “en”: “001” , “nl”:010”, “it”:”100}

<br/>
The trainer object makes a list of vectorized data inputs with the help of the Language_Features class for use in training . It also adds the inputs to the train list one language at a time in order and maintaining the order . This is important for proper training of the network . eg [ English, Dutch, Italian, English ,Dutch, Italian ......]
Neural_Network is instantiated specifying the number of inputs and outputs. It contains methods for feed­forward and back propagation operations.


An update_weights method is used while testing to initialize the network with the best weights. Test takes in the feature list and the input text and outputs the predicted language


<b>Training</b>


The network was trained on the entire data and after repeated experiments , the network was found to learn its best with a data set of 1200 articles and a learning rate of 1.0 for 10000 epochs. The data was trained with 100, 1000 and 10000 epochs with varying input sizes (400, 600 and 1200).

<br/>

<b>￼Testing</b>

<br/>
The network was first tested with the main.py file where the best weights were obtained after repeated testing and printing out the weights.

<br/>

The test.py was then updated with the best weights and used for final testing.
output_index_codes dictionary contains the mapping for the index of the languages in the output vector.
