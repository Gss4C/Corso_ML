#Stochastic GD
import numpy as np
class AdalineSGD(object):
    def __init__(self , eta=0.01 , n_iter = 10 , shuffle = True , random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False 
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self , X , y):
        self._initialize_weights (X.shape[1]) #shape dà numero di righe o colonne
        self.cost_=[]
        for i in range(self.n_iter):
            if self._shuffle: 
                X,y = self._shuffle(X,y)
            cost = []
            for xi , target in zip(X,y):
                cost.append(self._update_weights(xi , target))
            avg_cost=sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    
    def _shuffle(self , X , y): #Funzione che fa lo shuffle
        r=self.rgen.permutation(len(y))
        return X[r] , y[r]
    
    def _update_weights(self , xi , target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:]+=self.eta*xi.dot(error)
        self.w_[0]+= self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def partial_fit(self , X , y): #serve per il caso nel quale ho i pesi inizializzati
        if not self.w_initialized : 
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0]>1:
            for xi , target in zip(X , y):
                self._update_weights(xi, targer)
        else: #Mi serve per non far avere errore a zip se ho 1 solo valore
            self._update_weights(X,y)
        return self
    
    def _initialize_weights(self , m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_=self.rgen.normal(loc = 0.0 , scale = 0.01 , size=1+m)
        self.w_initialized = True
    
    def net_input(self , X):
        return np.dot(X , self.w_[1:])+self.w_[0]
    
    def activation(self , X):
        return X
    
    def predict(self , X):
        return np.where(self.activation(self.net_input(X))>= 0.0 , 1 , -1)
#-----------------#
#Adaline
import numpy as np
class AdalineGD(object):
    def __init__ ( self , eta=0.01, n_iter =50, random_state=1):
        self.eta = eta 
        self.n_iter = n_iter 
        self.random_state = random_state
    
    def fit(self , X , y):
        rgen = np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0 , scale=0.01 , size=1+X.shape[1])
        self.cost_=[]
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input) 
            errors =(y-output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost= (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self , X):
        return np.dot(X , self.w_[1:]) + self.w_[0]
    def activation(self , X):
        return X
    def predict(self , X):
        return np.where(self.activation(self.net_input(X))>=0.0,1,-1)