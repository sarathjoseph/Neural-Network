
class Language_Features(object):
    
    def __init__(self,features):
        
        self.feature_list = features
     
        
    def vectorize(self, condition, text):
        """ Returns the proportions of words in text with the condition """
        
        count = 0
        words = text.split(" ")
        
        for word in words:
            if len(word)<1:continue
            if condition(word):count += 1
            
        return float(count) / len(words)
            