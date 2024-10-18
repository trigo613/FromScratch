import numpy as np
from numba import njit

class Layer:
    def __init__(self):
        self.input_data = None 
        self.output_data = None
        self.is_parameterized = False
    def forward(self,X):
        pass

    def predict(self,X):
        return self.forward(X)
    
    def compute_gradient(self,output_error):
        pass
    def apply_gradient(self,learning_rate,gradients=[]):
        pass
    def get_gradients(self):
        return []

    def backward(self,output_error):
        pass
        
    def get_regularization_weights(self):
        return []

    def get_regularization_gradients(self):
        return []

class Activation(Layer):
    def __init__(self):
        super().__init__()
        
    def activation(self, X):
        raise NotImplementedError("Function must be implemented in subclass")

    def d_activation(self, X):
        raise NotImplementedError("Function must be implemented in subclass")

    def apply_gradient(self,learning_rate,gradients=[]):
        pass

    def forward(self, X):
        self.input_data = X
        self.output_data = self.activation(X)  # This should correctly update output_data in child class
        return self.output_data

    def backward(self,output_error):
        return self.d_activation(self.input_data) * output_error

    def compute_gradient(self, output_error):
        input_error = self.d_activation(self.input_data) * output_error
        return {'input_error': input_error, 'gradients': []}

class Linear(Layer):
    def __init__(self,input_size,output_size,weight_initialization='he'):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.initialize_weights(weight_initialization)
        
        self.dW = np.ones(shape=(self.input_size,self.output_size))
        self.db = np.ones(shape=(1,self.output_size))

        self.is_parameterized = True

    def initialize_weights(self,init):
        if init == 'random':
            self.W = np.random.uniform(size=(self.input_size,self.output_size))-0.5
        elif init == 'xavier': #Good for tanh
            self.W = np.random.randn(self.input_size, self.output_size) * np.sqrt(1 / self.input_size)
        elif init == 'xavierV2': #Good for tanh
            self.W = np.random.randn(self.input_size, self.output_size) * np.sqrt(2 / (self.input_size+self.output_size))
        elif init == 'he': #Good for relu
            self.W = np.random.randn(self.input_size, self.output_size) * np.sqrt(2 / self.input_size)
        self.b = np.zeros((1,self.output_size))



    def forward(self,X):
        self.input_data = X
        self.output_data = np.dot(X,self.W) + self.b
        return self.output_data

    def get_gradients(self):
        return [self.dW,self.db]

    def backward(self,output_error):
        self.dW = np.dot(self.input_data.transpose(),output_error)
        self.db = np.sum(output_error,axis=0,keepdims=True)
        return np.dot(output_error,self.W.transpose()) 

    def get_regularization_weights(self):
        return [self.W]

    def get_regularization_gradients(self):
        return [self.dW]

    def apply_gradient(self,learning_rate,gradients):
        dW = gradients[0]
        db = gradients[1]
        self.W = self.W - (learning_rate*dW)
        self.b = self.b - (learning_rate*db)

class BatchNormalization(Layer):
    def __init__(self, input_size):
        super().__init__()
        
        self.input_size = input_size
        self.X_norm = None
        self.gamma = np.ones((1, self.input_size))
        self.beta = np.zeros((1, self.input_size))

        self.dgamma = np.zeros((1, self.input_size))
        self.dbeta = np.zeros((1, self.input_size))
        self.average_list_size = 100
        
        self.list_counter = 0
        self.mew = np.zeros((1, self.input_size))
        self.mew_list = np.zeros((self.average_list_size, self.input_size))

        self.sigma_squared = np.zeros((1, self.input_size))
        self.sigma_squared_list = np.zeros((self.average_list_size, self.input_size))

        self.epsilon = 1e-15
        self.is_parameterized = True
    def forward(self, X):
        self.input_data = X
        n = len(X)

        self.mew = np.mean(X, axis=0, keepdims=True)
        self.sigma_squared = (1/n) * np.sum((X - self.mew) ** 2, axis=0, keepdims=True)
        self.list_counter = (self.list_counter + 1) % self.average_list_size
        self.sigma_squared_list[self.list_counter] = self.sigma_squared
        self.mew_list[self.list_counter] = self.mew

        self.denominator = np.sqrt(self.sigma_squared + self.epsilon)
        self.X_norm = (X - self.mew) / self.denominator
        self.output_data = self.gamma * self.X_norm + self.beta 
        return self.output_data
    def predict(self, X):
        sigma_squared = np.mean(self.sigma_squared_list, axis=0, keepdims=True)
        mew = np.mean(self.mew_list, axis=0, keepdims=True)
        X_norm = (X - mew) / np.sqrt(sigma_squared + self.epsilon)
        return self.gamma * X_norm + self.beta
    def get_gradients(self):
        return [self.dgamma, self.dbeta]
    def backward(self, output_error):
        self.dbeta = np.sum(output_error, axis=0, keepdims=True)
        self.dgamma = np.sum(self.X_norm * output_error, axis=0, keepdims=True)

        n = len(self.input_data)
        dX_norm = output_error * self.gamma
        X = self.input_data
        nominator = X - self.mew

        d_nominator = 1  # Derivative of (X - self.mew) with respect to X is 1
        d_denominator = 0.5 * (1 / self.denominator) * (2 * nominator / n)

        input_error = dX_norm * (d_nominator * self.denominator - nominator * d_denominator) / (self.denominator ** 2)

        return input_error
    def get_regularization_weights(self):
        return [self.gamma, self.beta]
    def get_regularization_gradients(self):
        return [self.dgamma, self.dbeta]
    def apply_gradient(self, learning_rate, gradients):
        dgamma, dbeta = gradients
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

class Dropout(Layer):
    def __init__(self,keep_probability=0.5):
        super().__init__()
        self.keep_probability = keep_probability
        self.D = None

    def forward(self,X):
        self.D = np.random.uniform(size=X.shape) < self.keep_probability
        out = (X*self.D)/self.keep_probability
        return out

    def backward(self,output_error):
        return self.D*output_error/self.keep_probability
        
        
    def compute_gradient(self,output_error):
        input_error = self.D * output_error
        return {'input_error': input_error, 'gradients': []}

    def predict(self,X):
        return X

class FlattenConv(Layer):
    def __init__(self):
        super().__init__()
        self.n_samples = 0
        self.n_channels = 0
        self.n_rows = 0
        self.n_cols = 0

    def forward(self,X):
        self.n_samples,self.n_channels,self.n_rows,self.n_cols = X.shape
        rest = self.n_channels*self.n_rows*self.n_cols
        return X.reshape((self.n_samples,rest))

    def backward(self,output_error):
        return output_error.reshape((self.n_samples,self.n_channels,self.n_rows,self.n_cols))

class ReLU(Activation):
    def __init__(self):
        super().__init__()
        
    def activation(self,X):
        return X*(X>0)

    def d_activation(self,X):
        return (X>0).astype(int)

class Sigmoid(Activation):        
    def __init__(self):
        super().__init__()
        
    def activation(self, X):
        return 1 / (1 + np.exp(-X))

    def d_activation(self, X):
        s = self.output_data
        return s * (1 - s)

class Tanh(Activation):
    def __init__(self):
        super().__init__()
        
    def activation(self,X):
        return np.tanh(X)
        
    def d_activation(self,X):
        t = self.output_data
        return  1. -(t**2)

class NeuralNetwork:
    def __init__(self,layers):
        self.layers = layers

    def forward(self,X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self,X):
        output = X
        for layer in self.layers:
            output = layer.predict(output)
        return output

class Regularization:
    def __init__(self,lambd = 0.01):
        self.lambd = lambd
    def get_weight_penalty(self,W,n):
        raise NotImplementedError("Function must be implemented in subclass")
    def get_weight_gradient(self,W,n):
        raise NotImplementedError("Function must be implemented in subclass")

class ZeroRegularization(Regularization):
    def __init__(self,lambd = 0.01):
        super().__init__(lambd)
    def get_weight_penalty(self,W,n):
        return 0
    def get_weight_gradient(self,W,n):
        return np.zeros_like(W)

class L2Regularization(Regularization):
    def __init__(self,lambd = 0.01):
        super().__init__(lambd)
    def get_weight_penalty(self,W,n):
        return (self.lambd/(2*n))*np.sum(W**2)
    def get_weight_gradient(self,W,n):
        return (self.lambd/n)*W

class L1Regularization(Regularization):
    def __init__(self,lambd = 0.01):
        super().__init__(lambd)
    def get_weight_penalty(self,W,n):
        return  (self.lambd/n)*np.sum(np.abs(W))
    def get_weight_gradient(self,W,n):
        return (self.lambd/n)*((W>0).astype(int) - 0.5)*2

class Loss:
    def __init__(self,layers = None,regularization=ZeroRegularization()):
        self.layers = layers
        self.parameterized_layers = [l for l in layers if l.is_parameterized]
        self.regularization = regularization
        self.error = 0
        self.y_true = 0
        self.y_pred = 0

    def backwards(self):
        output_error = self.compute_gradient(self.y_true,self.y_pred)
        for i in range(len(self.layers)-1,-1,-1):
            layer = self.layers[i]
            output_error = layer.backward(output_error)
            
            layer_regularization_weights = layer.get_regularization_weights()
            layer_gradients = layer.get_regularization_gradients()
            for j in range(len(layer_regularization_weights)):
                weight_gradient = self.regularization.get_weight_gradient(layer_regularization_weights[j],len(self.y_true))
                layer_gradients[j] += weight_gradient
    
    def compute_gradient(self,y_true,y_pred):
        pass
    
    def compute_loss(self,y_true,y_pred):
        pass
        
    def loss(self,y_true,y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.error = self.compute_loss(y_true,y_pred) 
        penalty = 0
        for layer in self.parameterized_layers:
            for weight in layer.get_regularization_weights():
                penalty += self.regularization.get_weight_penalty(weight,len(self.y_true))
        return self.error + penalty

class MSELoss(Loss):
    def __init__(self,layers = None,regularization=ZeroRegularization()):
        super().__init__(layers,regularization)
    def mse(self,y_true,y_pred):
        out = y_pred - y_true
        out = out**2
        return np.mean(out)
    def compute_gradient(self,y_true,y_pred):
        out = (2/len(y_true))*(y_pred-y_true)
        return out
    def compute_loss(self,y_true,y_pred):
        return self.mse(y_true,y_pred)

class CELoss(Loss):
    def __init__(self,layers = None,regularization=ZeroRegularization(),epsilon=1e-15,apply_softmax=True):
        super().__init__(layers,regularization)
        self.epsilon = epsilon
        self.apply_softmax = apply_softmax

    def softmax(self,X):
        eX = np.exp(X)
        denominator = np.sum(eX,axis=1,keepdims=True)
        return eX/(denominator+self.epsilon)

    def _clip_probabilities(self, y_pred):
        return np.clip(y_pred, self.epsilon, 1 - self.epsilon)

    def compute_loss(self, y_true, y_pred):
        if self.apply_softmax:
            y_pred = self.softmax(y_pred)
        y_pred = self._clip_probabilities(self.y_pred)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_gradient(self, y_true, y_pred):
        y_pred = self._clip_probabilities(y_pred)
        if self.apply_softmax:
            return y_pred-y_true
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

class LearninRateDecay:
    def __init__(self,optimizer):
        self.epoch = 0
        self.optimizer = optimizer
    def step(self):
        pass

class InverseTimeDecay(LearninRateDecay):
    def __init__(self,optimizer,decay=0.01):
        super().__init__(optimizer)
        self.decay=decay
    def step(self):
        self.optimizer.lr = self.optimizer.lr/(1+self.decay*self.epoch)
        self.epoch += 1

class ExponentialDecay(LearninRateDecay):
    def __init__(self,optimizer,base = 0.99):
        super().__init__(optimizer)
        self.base=base
    def step(self):
        self.optimizer.lr = self.optimizer.lr*(self.base**self.epoch)
        self.epoch += 1

class Optimizer:
    def __init__(self,layers,lr=0.01):
        self.lr = lr
        self.layers = [l for l in layers if l.is_parameterized]
        self.final_gradients = {i:self.layers[i].get_gradients() for i in range(len(self.layers))}

    def calculate_final_gradients(self):
        raise NotImplementedError("Function must be implemented in subclass")

    def step(self):
        self.calculate_final_gradients()
        for i in range(0,len(self.layers)):
            self.layers[i].apply_gradient(self.lr,self.final_gradients[i])

class SimpleOptimizer(Optimizer):
    def __init__(self,layers,lr=0.01):
        super().__init__(layers,lr)

    def calculate_final_gradients(self):
        for i in range(len(self.layers)-1,-1,-1):
            self.final_gradients[i] = self.layers[i].get_gradients()

class RMSProp(Optimizer):
    def __init__(self,layers,lr=0.01,beta=0.9):
        super().__init__(layers,lr)
        self.beta = beta
        self.epsilon = 1e-8
        self.S_gradients = {i:self.layers[i].get_gradients() for i in range(len(self.layers))}
        
    def calculate_final_gradients(self):
        for i in range(len(self.layers)-1,-1,-1):
            gradients = self.layers[i].get_gradients()
            for j in range(len(gradients)):
                self.S_gradients[i][j] = self.beta*self.S_gradients[i][j] + (1-self.beta)*(gradients[j]**2)
                self.final_gradients[i][j] = gradients[j]/((np.sqrt(self.S_gradients[i][j])) + self.epsilon)

class Adam(Optimizer):
    def __init__(self,layers,lr=0.01,beta_s=0.999,beta_v=0.9):
        super().__init__(layers,lr)
        
        self.beta_s = beta_s
        self.beta_v = beta_v
        
        self.t = 0
        
        self.epsilon = 1e-8
        
        self.S_gradients = {i:self.layers[i].get_gradients() for i in range(len(self.layers))}
        self.V_gradients = {i:self.layers[i].get_gradients() for i in range(len(self.layers))}

        
    def calculate_final_gradients(self):
        self.t+=1
        for i in range(len(self.layers)-1,-1,-1):
            gradients = self.layers[i].get_gradients()
            for j in range(len(gradients)):
                
                self.S_gradients[i][j] = (self.beta_s*self.S_gradients[i][j]) + ((1-self.beta_s)*(gradients[j]**2))
                corrected_s = self.S_gradients[i][j]/(1-(self.beta_s**self.t))

                self.V_gradients[i][j] = (self.beta_v*self.V_gradients[i][j]) + ((1-self.beta_v)*gradients[j])
                corrected_v = self.V_gradients[i][j]/(1-(self.beta_v**self.t))

                self.final_gradients[i][j] = corrected_v/(np.sqrt(corrected_s) + self.epsilon)


@njit
def apply_cross_correlation_3d(X,kernel,output):
    n_channels,ker_rows,ker_cols = kernel.shape
    output_rows,output_cols = output.shape
    for i in range(output_rows):
        for j in range(output_cols):
            output[i][j] = np.sum(X[:,i:i+ker_rows,j:j+ker_cols]*kernel)

@njit
def apply_cross_correlation_2d(X,kernel,output):
    ker_rows,ker_cols = kernel.shape
    output_rows,output_cols = output.shape
    for i in range(output_rows):
        for j in range(output_cols):
            output[i][j] = np.sum(X[i:i+ker_rows,j:j+ker_cols]*kernel)

@njit
def apply_full_cross_correlation_2d(X, kernel, output):
    in_rows, in_cols = X.shape
    ker_rows, ker_cols = kernel.shape
    output_rows, output_cols = output.shape

    rows_p = ker_rows - 1
    cols_p = ker_cols - 1
    padded_rows = in_rows + 2 * rows_p
    padded_cols = in_cols + 2 * cols_p
    
    padded_x = np.zeros((padded_rows, padded_cols))
    padded_x[rows_p:rows_p + in_rows, cols_p:cols_p + in_cols] = X
    
    for i in range(output_rows):
        for j in range(output_cols):
            subarray = padded_x[i:i + ker_rows, j:j + ker_cols]
            output[i, j] = np.sum(subarray * kernel)


@njit
def apply_kernels(X,kernels,output):
    #output_shape = (n_kernels,output_rows,output_cols)
    n_kernels,ker_channels,ker_rows,ker_cols = kernels.shape
    n_kernels,output_rows,output_cols = output.shape
    
    for i in range(n_kernels):
        apply_cross_correlation_3d(X,kernels[i],output[i])

@njit
def apply_kernels_to_batch(X,kernels,biases,full=False):
    #input_shape = (n_images,n_chanels,n_rows,n_cols)
    #kernels_shape = (n_kernels,n_chanels,n_rows,n_cols)
    #output_shape = (n_images,n_kernels,output_rows,output_cols)
    n_images,in_channels,in_rows,in_cols = X.shape
    n_kernels,ker_channels,ker_rows,ker_cols = kernels.shape
    assert(in_channels==ker_channels)
    if not full:
        output_rows = in_rows-ker_rows+1
        output_cols = in_cols-ker_cols+1
        output_shape = (output_rows,output_cols)
        output = np.zeros((n_images,n_kernels,output_rows,output_cols))
        padded_x = X
        
    else:  # full == True
        rows_p = ker_rows - 1
        cols_p = ker_cols - 1
        padded_rows = in_rows + 2 * rows_p
        padded_cols = in_cols + 2 * cols_p        
        padded_x = np.zeros((n_images, in_channels, padded_rows, padded_cols))        
        padded_x[:, :, rows_p:-rows_p, cols_p:-cols_p] = X
        output_rows = padded_rows - ker_rows + 1
        output_cols = padded_cols - ker_cols + 1
        output_shape = (output_rows, output_cols)
        output = np.zeros((n_images, n_kernels, output_rows, output_cols))

    for i in range(n_images):
        apply_kernels(padded_x[i],kernels,output[i])
    
    output += biases
    return output

@njit
def rotate_kernels_180(kernels):
    num_kernels, input_channels, kernel_size, _ = kernels.shape
    rotated_kernels = np.zeros_like(kernels)
    
    for k in range(num_kernels):
        for c in range(input_channels):
            rotated_kernels[k, c] = np.flipud(np.fliplr(kernels[k, c]))
    
    return rotated_kernels

@njit
def conv_backward(output_error, X, kernels, biases):
    # X_shape = (n_images, n_channels, n_rows, n_cols)
    # kernels_shape = (n_kernels, n_channels, ker_rows, ker_cols)
    # biases_shape = (n_kernels, 1, 1)
    # output_error_shape = (n_images, n_kernels, out_rows, out_cols)
    
    n_kernels, n_channels, ker_rows, ker_cols = kernels.shape
    n_images, output_channels, output_rows, output_cols = output_error.shape
    
    dB = np.zeros(biases.shape)

    #temp = np.sum(output_error, axis=(0, 2, 3)) #(64,) -> #(64,1,1)
    for i in range(n_channels):
        dB[i][0][0] = np.sum(output_error[:,i,:,:])

    
    dK = np.zeros_like(kernels)
    t = np.zeros(dK[0,0].shape)
    for i in range(n_kernels):
        for c in range(n_channels):
            for j in range(n_images):
                # Slice the relevant part of output_error and input
                relevant_error = output_error[j, i]  # Shape: (output_rows, output_cols)
                relevant_input = X[j, c]  # Shape: (input_rows, input_cols)
                apply_cross_correlation_2d(relevant_input,relevant_error,t)
                dK[i,c] += t

    # Gradient with respect to the input (dX)
    rotated_kernels = rotate_kernels_180(kernels)  # Rotate kernels by 180 degrees
    dX = np.zeros_like(X)
    t = np.zeros(dX[0][0].shape)
    for i in range(n_images):
        for c in range(n_channels):
            # For each input channel, compute the convolution of the output error with the rotated kernels
            relevant_kernels = rotated_kernels[:, c]  # Shape: (n_kernels, ker_rows, ker_cols)
            for k in range(n_kernels):
                # Apply the rotated kernel to the output error
                apply_full_cross_correlation_2d(
                       output_error[i, k],
                       relevant_kernels[k],
                        t)
                
                dX[i, c] += t

    return dK, dB, dX

@njit 
def max_pool_forward(X,pool_size):
    #X shape = (n_images,n_channels,n_rows,n_cols)
    #output_shape = (n_images,n_channels,n_rows/pool_size,n_cols/pool_size)
    in_images,in_channels,in_rows,in_cols = X.shape
    output_rows,output_cols  = int(in_rows/pool_size),int(in_cols/pool_size)
    output = np.zeros((in_images,in_channels,output_rows,output_cols))
    mask = np.zeros(X.shape)
    
    for n in range(in_images):
        for c in range(in_channels):
            for i in range(output_rows):
                i_p = i*pool_size
                for j in range(output_cols):
                    j_p = j*pool_size
                    curr_image = X[n,c,i_p:i_p+pool_size,j_p:j_p+pool_size] 
                    argmax = curr_image.argmax()
                    max_row,max_col = argmax//pool_size,argmax%pool_size
                    output[n,c,i,j] = curr_image[max_row,max_col]
                    mask[n,c,i_p+max_row,j_p+max_col] = 1

    return output,mask

@njit 
def max_pool_backward(output_error,mask,pool_size):
    input_error = mask.copy()
    
    images,channels,out_rows,out_cols = output_error.shape
    
    for n in range(images):
        for c in range(channels):
            for i in range(out_rows):
                i_p = i*pool_size
                for j in range(out_cols):
                    j_p = j*pool_size
                    max_item = output_error[n,c,i,j]
                    
                    input_error[n,c,i_p:i_p+pool_size,j_p:j_p+pool_size]  *= max_item

    return input_error
    
class MaxPool2D(Layer):
    def __init__(self, pool_size):
        super().__init__()
        self.mask = None
        self.pool_size = pool_size
        
    def forward(self, X):
        self.input_data = X
        self.output_data,self.mask = max_pool_forward(X,self.pool_size)
        return self.output_data

    def backward(self, output_error):
        input_error = max_pool_backward(output_error,self.mask,self.pool_size)
        return input_error

class Conv2D(Layer):
    def __init__(self, input_shape, kernel_size, num_kernels):
        super().__init__()
        self.input_channels, self.input_height, self.input_width = input_shape
        
        self.input_shape = input_shape

        self.is_parameterized = True

        self.kernel_size = kernel_size
        
        self.num_kernels = num_kernels
                
        self.kernels_shape = (num_kernels, self.input_channels, kernel_size, kernel_size)
        
        self.kernels = np.random.randn(*self.kernels_shape) * np.sqrt(2. / (self.input_channels * self.kernel_size**2))
            
        self.biases = np.random.random((self.num_kernels,1,1))

        self.dkernels = np.ones(self.kernels_shape)
        
        self.dbiases = np.ones(self.biases.shape)


    def forward(self, X):
        self.input_data = X
        self.output_data = apply_kernels_to_batch(X,self.kernels,self.biases,full=False)
        return self.output_data

    def backward(self, output_error):
        # print(output_error)
        # print("------------------")
        dK,dB,dX = conv_backward(output_error,self.input_data,self.kernels,self.biases)
        #print(dK,dB,dX)
        self.dbiases = dB
        self.dkernels = dK
        return dX
    
        
    def get_gradients(self):
        return [self.dkernels,self.dbiases]
        
    def get_regularization_weights(self):
        return [self.kernels,self.biases]

    def get_regularization_gradients(self):
        return [self.dkernels,self.dbiases]

    def apply_gradient(self,learning_rate,gradients):
        dkernels = gradients[0]
        dbiases = gradients[1]
        self.kernels = self.kernels - (learning_rate*dkernels)
        self.biases = self.biases - (learning_rate*dbiases)
        