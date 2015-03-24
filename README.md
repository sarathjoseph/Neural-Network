# Neural-Network


A Neural Network for Language Identification (Eng, Dutch , Italian ) using Python
Wikipedia Language Detection Feed­forward neural network with std backpropagation implementation

Features
● Proportion of words starting with ‘v’ and containing the bigram “ij”
● Proportion of words ending with ‘i’ and ending with ‘o’ (length>3) and ending with ‘a’
(length>1)
● Proportion of words containing the bigram “jk”
● Proportion of words containing the bigram “th” and starting and ending with t and
words containing the trigram ‘tio’ and the bigram ‘fo’.
Rationale
● Dutch contains frequent bigrams as “ij” and “jk” and have a good proportion of words starting with v.
● Italian has many words ending with vowels and there are very few words in English and Dutch that end with i. Words ending with a and o with length greater than 1 and 3 respectively are considered since English has common words like “so” and “to”.
● English has the bigram “th” appearing almost in every sentence with frequent words like with, that, this , there, the , than etc. Many words also start and end with ‘t’. The trigram ‘tio’ appears in many English words frequently as does the bigram ‘fo’.
Data Gathering
Data was gathered using python scripts and the WIKI random article generator feature. Around 1200 unique articles (400 each for all three languages) were obtained.
Scripts were run on downloaded articles to weed out duplicates using file hash.
The articles were downloaded into folders for the respective languages.
en , nl and it.
￼Design
Classes
● Neural_Network
● Network_Trainer
● Language_Features
Language_Features class is instantiated with the feature set and contains the vectorize method which turns the features into real numbers. The feature set is passed in as callbacks for allowing addition of features dynamically to the Language Features class.
Network_Trainer class is instantiated with the language features object , the codes and the input folder for training. The codes are kept in a dictionary containing the mapping between folder names (language names) and output codes
eg { “en”: “001” , “nl”:010”, “it”:”100}
The trainer object makes a list of vectorized data inputs with the help of the Language_Features class for use in training . It also adds the inputs to the train list one language at a time in order and maintaining the order . This is important for proper training of the network . eg [ English, Dutch, Italian, English ,Dutch, Italian ......]
Neural_Network is instantiated specifying the number of inputs and outputs. It contains methods for feed­forward and back propagation operations.
An update_weights method is used while testing to initialize the network with the best weights. Test takes in the feature list and the input text and outputs the predicted language
Training
The network was trained on the entire data and after repeated experiments , the network was found to learn its best with a data set of 1200 articles and a learning rate of 1.0 for 10000 epochs. The data was trained with 100, 1000 and 10000 epochs with varying input sizes (400, 600 and 1200).
￼Testing
The network was first tested with the main.py file where the best weights were obtained after repeated testing and printing out the weights.
The test.py was then updated with the best weights and used for final testing.
The neural network also has a method called output_index_codes which contains the mapping for the index of the languages in the output vector.
Observations
● The network predicts the best when given an input in excess of 20 words
● Italian and Dutch text are predicted better than English text
CODE EXECUTION
Lab2.zip contains
● Code Folder ● data Folder ● Report File
The Code folder contains test.py file which needs to be executed as $ python test.py < text input>
Example usage
$ python test.py ontrollo militare di Aidid, tramite gli UH­60 Black Hawk che
NOTE: All the other files in the folder are needed (especially nn.py and lang.py for proper execution)
