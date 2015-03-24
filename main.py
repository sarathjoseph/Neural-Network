# -*- coding: utf-8 -*- 
from lang import Language_Features
from nn import Neural_Network
from trainer import Network_Trainer


if __name__ == "__main__":
    
    text="Welcome to Neural Networks"
     
    """ The codes should have keys as the language folder names"""
    codes = {"en": [0, 0, 1] , "nl":[0, 1, 0], "it":[1, 0, 0]}
    
    features = [
                  lambda w:w[ 0].lower() == 'v' or "ij" in w.lower(),  # words that start with v or contain the bigram "ij"
                  lambda w:w[-1].lower() == 'i' or (w[-1].lower()=='a' and len(w)>1) or (w[-1].lower()=='o' and len(w)>3),  # words that end with i
                  lambda w:w[ 0].lower() == "jk" in w.lower() or (w[0].lower()=='d' and len(w)<3),  # words that start with d
                  lambda w:w[0].lower()=='t'or w[-1].lower() == 't'or "th" in w.lower() or "tio" in w.lower() or "fo" in w.lower()]  # words that contain the bigram "th" and the trigram "tio"
    
    """ Initialize class with features """
    lf = Language_Features(features)
    
    folder = "/home/joseph/Desktop/AI Lab2/complete"  
    
    ntrain = Network_Trainer(lf, folder, codes)
       
    nn = Neural_Network(4,3)
    weights={'input_hidden_bias': [2.948343288918918, 1.1632824615810022, 2.8471729566158706, -1.307920045405036, 2.9751354452386334], 'input_hidden': [[0.9504649081548636, -55.20044085145969, 0.5344791876828107, -93.80240258774394, 1.0182446277192532], [1.849047805911803, 38.95430746008158, -1.3181774889334184, 67.80762919398143, 0.6898168781791706], [0.7771636523937926, 74.02090943452536, 0.4781634439066883, -197.8024853428895, 0.2694675348604373], [0.20003232651157307, -54.00816921079571, 2.3865793299093068, 49.72797514928896, 0.4681677743814763]], 'hidden_output bias': [-3.8351163121616376, 0.5941282055619822, -0.7312277733157507], 'hidden_output': [[-2.440636534080138, 0.15158336019262922, -2.1322829501582645], [7.353423447626104, -2.863202016388184, -60.5887152938032], [-5.085840148056722, 0.9446032215586617, 0.1806830638490392], [27.4863786873561, -6.220083144313221, 7.203167268989403], [-3.617994115141314, 0.9183837726126198, -1.375089269779633]]}

    nn.update_weights(weights)
    nn.train(ntrain,5,1.0)  # pass in train object, no of epochs and learning rate
    nn.output_index_codes({0:"Italian", 1:"Dutch", 2:"English"})
    
    
    print(nn.get_weights())
    print(nn.test(lf,text))