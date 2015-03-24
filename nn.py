import math
import random

class Neural_Network(object):

    def __init__(self, num_inputs, num_outputs):
        
        self.n_inputs = num_inputs
        self.n_outputs = num_outputs
        self.inputs = [0 for _ in range(num_inputs)] 
        self.outputs = [0 for _ in range(num_outputs)]
        self.n_hln = num_inputs + 1  # number of hidden layer neurons

        self.ihb_w = [random.random() for _ in range(self.n_hln) ]  # input hidden bias weights
        self.hob_w = [random.random() for _ in range(self.n_outputs) ]  # hidden output bias weights 
        
        self.ih_w = [[random.random() for _ in range(self.n_hln)] for _ in range(num_inputs)]  # input hidden weights
        self.ho_w = [[random.random() for _ in range(num_outputs)]for _ in range(self.n_hln) ]  # hidden_output_weights
    
    def output_index_codes(self,codes):
        self.index_codes=codes
        
    def update_weights(self, weights):
        
        self.ihb_w = weights["input_hidden_bias"]
        self.hob_w = weights["hidden_output bias"]
        self.ih_w = weights["input_hidden"]
        self.ho_w = weights["hidden_output"]
        
    
    def get_weights(self):
        
        return {"input_hidden_bias"  :self.ihb_w, "hidden_output bias" :self.hob_w, "input_hidden" :self.ih_w, "hidden_output":self.ho_w }            
        
    def feed_forward(self, d):
     
        outputs_h = [0.0] * self.n_hln
        for neuron in range(self.n_hln):
            output = 0.0
            for i, _input in enumerate(d):
                output += self.ihb_w[neuron] * 1 + _input * self.ih_w[i][neuron]
       
            outputs_h[neuron] = self.sigmoid(output)
            
        out = [0.0] * self.n_outputs
        self.outputs_h = outputs_h  # for access in backpropogatation
        
        for neuron in range(self.n_outputs):
            output = 0.0
            for i, _input in enumerate(outputs_h):
                output += self.hob_w[neuron] * 1 + _input * self.ho_w[i][neuron]

            out[neuron] = self.sigmoid(output)
        
        return out
        
    def back_propogate(self, out, example):
        
        target = example[-1]
        # Compute error of output neurons    
        out_error = [out[i] * (1 - out[i]) * (target[i] - out[i]) for i in range(self.n_outputs)]
        
        sum([abs(e) for e in out_error])
            
        #  new_ho_w = [[0.0 for _ in range(self.n_outputs)]] * (self.n_hln)
        hidden_errors = [0.0] * self.n_hln
        
        for j in range(self.n_outputs):  # change bias weights for output layer
            self.hob_w[j] += self.lnr * out_error[j] * 1     
         
        # Calculate new output layer weights and hidden layer errors   
        for i in range(self.n_hln):
            for j in range(self.n_outputs):
                self.ho_w[i][j] = self.ho_w[i][j] + self.lnr * self.outputs_h[i] * out_error[j]
                hidden_errors[i] += (out_error[j] * self.ho_w[i][j]) * (self.outputs_h[i]) * (1 - self.outputs_h[i])  # using old weights here
        
        # Change hidden layer weights 
        
        for i in range(self.n_inputs):
            for j in range(self.n_hln):
                self.ih_w[i][j] += self.lnr * hidden_errors[j] * example[i]
                
        for j in range(self.n_hln):  # change bias weights for hidden layer
            self.ihb_w[j] += self.lnr * hidden_errors[j] * 1        

        
        return sum([abs(e) for e in out_error])
        
    def train(self, train_obj, epochs, learning_rate):

        self.lnr = learning_rate

        train_set = train_obj.get_data()
        
        for _ in range(epochs):
            e=0.0
            for example in train_set:
                
                output = self.feed_forward(example[:-1])
                error=self.back_propogate(output, example)
                e+=error
            
            #error_epoch=e/len(train_set)   
    
    def test(self, f_obj, text):
        vector = []
        for feature in f_obj.feature_list:
                    vector.append(f_obj.vectorize(feature, text))
        
        output=self.feed_forward(vector) 
        print(output)
        lang_index=output.index(max(output))
        return self.index_codes[lang_index]
                
    
    def sigmoid(self, x):
        
        return 1 / (1 + math.exp(-x))

        

