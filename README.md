# Neural-Network


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
