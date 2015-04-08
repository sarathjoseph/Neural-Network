# -*- coding: utf-8 -*- 

from lang import Language_Features
from nn import Neural_Network
from trainer import Network_Trainer


if __name__ == "__main__":
    
    text="Welcome to Neural Networks"
     
    """ The codes should have keys as the language folder names"""
    codes = {"en": [0, 0, 1] , "nl":[0, 1, 0], "it":[1, 0, 0]}
    
    features = [
                  lambda w:w[ 0].lower() == 'v'  or "ij" in w.lower(),  # words that start with v or contain the bigram "ij"
                  lambda w:w[-1].lower() == 'i'  or (w[-1].lower()=='a' and len(w)>1) or (w[-1].lower()=='o' and len(w)>3),  # words that end with i
                  lambda w:w[ 0].lower() == "jk" in w.lower() or (w[0].lower()=='d' and len(w)<3),  # words that start with d
                  lambda w:w[ 0].lower() == 't'  or w[-1].lower() == 't'or "th" in w.lower() or "tio" in w.lower() or "fo" in w.lower()]  # words that contain the bigram "th" and the trigram "tio"
    
    """ Initialize class with features """
    lf = Language_Features(features)
    
    folder = "/home/joseph/Desktop/AI Lab2/complete"  
    
    ntrain = Network_Trainer(lf, folder, codes)
       
    nn = Neural_Network(4,3)

    train_data=ntrain.get_data()


    nn.train(train_data,5,1.0)  # pass in train data, no of epochs and learning rate

    
    print(nn.get_weights())

    