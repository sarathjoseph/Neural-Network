# -*- coding: utf-8 -*- 
from nn import Neural_Network
from lang import Language_Features
import sys

if __name__ == "__main__":
    

    text = " ".join(sys.argv[1:])
    
    lf=Language_Features([
                  lambda w:w[ 0].lower() == 'v' or "ij" in w.lower(),  # words that start with v or contain the bigram "ij"
                  lambda w:w[-1].lower() == 'i' or (w[-1].lower()=='a' and len(w)>1) or (w[-1].lower()=='o' and len(w)>3),  # words that end with i
                  lambda w:w[ 0].lower() == "jk" in w.lower() or (w[0].lower()=='d' and len(w)<3),  # words that start with d
                  lambda w:w[0].lower()=='t'or w[-1].lower() == 't'or "th" in w.lower() or "tio" in w.lower() or "fo" in w.lower()]  # words that contain the bigram "th" and the trigram "tio"
    
                          ) # words that contain the bigram "th" and the trigram "tio"
    
    weights={'input_hidden_bias': [2.948343288918918, 1.1632824615810022, 2.8471729566158706, -1.307920045405036, 2.9751354452386334], 'input_hidden': [[0.9504649081548636, -55.20044085145969, 0.5344791876828107, -93.80240258774394, 1.0182446277192532], [1.849047805911803, 38.95430746008158, -1.3181774889334184, 67.80762919398143, 0.6898168781791706], [0.7771636523937926, 74.02090943452536, 0.4781634439066883, -197.8024853428895, 0.2694675348604373], [0.20003232651157307, -54.00816921079571, 2.3865793299093068, 49.72797514928896, 0.4681677743814763]], 'hidden_output bias': [-3.8351163121616376, 0.5941282055619822, -0.7312277733157507], 'hidden_output': [[-2.440636534080138, 0.15158336019262922, -2.1322829501582645], [7.353423447626104, -2.863202016388184, -60.5887152938032], [-5.085840148056722, 0.9446032215586617, 0.1806830638490392], [27.4863786873561, -6.220083144313221, 7.203167268989403], [-3.617994115141314, 0.9183837726126198, -1.375089269779633]]}
    #weights= {'input_hidden_bias': [9.842090160039051, -0.961296137357649, 3.409402215696614, 3.380926977667672, 3.3666376354775025], 'input_hidden': [[162.51031095011786, 183.97684841442037, 0.9015149722464426, 0.9409850639484273, 0.41544009672448223], [-143.28686494039803, 48.8178778676778, -0.0006221544850762926, 0.7786784450874774, 1.1880721140793475], [0.3884599739296217, 0.6594121235711843, 0.6480424902879466, 0.35174703302489874, 0.8617204358889209], [-27.149914490813174, -39.18718268278591, 0.6059083257696843, 1.2853325259250272, 0.5817686303263757]], 'hidden_output bias': [0.6793374838941008, -1.4043189028675382, -3.8289811640164793], 'hidden_output': [[-7.549806258355485, 9.067069808141579, 31.55882797568828], [-0.6252047341532094, 6.137821215792533, -268.799239082421], [0.4696321968546435, -0.43280575652930764, -3.7737442945149566], [0.7643199172076974, -0.9964293981249385, -3.7776009331380695], [0.434658152525398, -1.820643350688633, -4.547693903697286]]}

    nn = Neural_Network(4, 3)
    
    nn.update_weights(weights)

    index_codes={0:"Italian", 1:"Dutch", 2:"English"}
   
   # Testing 

    vector = [lf.vectorize(feature, text) for feature in lf.feature_list ]

    output=nn.test(vector)
     
    lang_index=output.index(max(output))

    # Print the identified language
    
    print(index_codes[lang_index])
          

    
 
