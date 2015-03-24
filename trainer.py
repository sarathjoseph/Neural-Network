    
from os import listdir as ls

class Network_Trainer(object):
    
    def __init__(self, feat, folder, codes):
        
        self.codes = codes
        self.book = {}
        self.train_set = []
        self.features=feat
        
        
        for item in ls(folder):
            self.book[item] = ls(folder + "/" + item)
            
        self.doc_count = len(self.book[self.codes.keys()[0]])
            
        for i in range(self.doc_count):
            for item in self.book:
                f = open(folder + "/" + item + "/" + self.book[item][i])
                text = f.read()
                vector = []
                for feature in feat.feature_list:
                    vector.append(feat.vectorize(feature, text))
                vector.append(self.codes[item])
                self.train_set.append(vector)
                f.close()
        
    def get_data(self):
        return self.train_set
    

        
        
    
