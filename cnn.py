"""
Fall 2022, 10-617
Assignment-2
Programming - CNN
TA in charge: Athiya Deviyani, Anamika Shekhar, Udaikaran Singh

IMPORTANT:
    DO NOT change any function signatures

Sept 2022
"""

from re import L
import numpy as np
import copy
import im2col_helper
import pickle



softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis = 1).reshape(-1,1)

def singleim2col(imgseq, k_height, k_width,outputH,outputW,padding,stride):
    # for the image, start with padding

    #
    ci = 0
    length = imgseq.shape[0]

    res = np.zeros((k_height*k_width, length*outputH*outputW))


    for h in range (outputH):
        # corresponding starting point on the image
        newh = stride*h
        for w in range (outputW):
            # corresponding starting point on the image
            neww = stride*w
            # store the corresponding window
            for n in range (length):
                img = imgseq[n]
                pad_img =np.pad(img, [(padding, ), (padding, )], mode='constant')


                res[:,ci] = pad_img[newh:newh+k_height,neww: neww+k_width].flatten()
                ci+=1

                
    return res

def im2col(X, k_height, k_width, padding=1, stride=1):
    '''
    Construct the im2col matrix of intput feature map X.
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    return a 2D array of shape (C*k_height*k_width, H*W*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) here, while in
    reality it can be other ways if it weren't for autograding tests.
    '''
    # read in all the parameters of the input 
    N = X.shape[0]
    C = X.shape[1]
    H = X.shape[2]
    W = X.shape[3]
    
    # calculate the output width and the output height
    outputH = (H - k_height + 2*padding)//stride+1
    outputW = (W - k_width + 2*padding)//stride+1
    result = np.zeros((C*k_height*k_width, outputH*outputW*N))

    for c in range (C):
        input = X[:,c,:,:]
        greyscale = singleim2col(input, k_height, k_width,outputH,outputW,padding,stride)

        result[c*k_height*k_width:(c+1)*k_height*k_width] = greyscale

    return result





def matchback(k_height, k_width,outputH,outputW,padding,stride,greyscale,N,H,W):
    ci = 0
    length = N
    res = np.zeros((N,H+2*padding,W+2*padding))
    for h in range (outputH):
        # corresponding starting point on the image
        newh = stride*h
        for w in range (outputW):
            # corresponding starting point on the image
            neww = stride*w
            # store the corresponding window
            for n in range (length):




                source = greyscale[:,ci]

                s = source.reshape((k_height,k_width))
                res[n,newh:newh+k_height,neww: neww+k_width] += s

                ci+=1

    res = res[:,padding:H+padding, padding:W+padding]          
    return res



def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    '''
    Map gradient w.r.t. im2col output back to the feature map.
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape
    '''
    N = X_shape[0]
    C = X_shape[1]
    H = X_shape[2]
    W = X_shape[3]
    
    # calculate the output width and the output height
    outputH = (H - k_height + 2*padding)//stride+1
    outputW = (W - k_width + 2*padding)//stride+1
    X = np.zeros((N,C,H,W))
    for c in range (C):
        greyscale = grad_X_col[c*k_height*k_width:(c+1)*k_height*k_width]

        target = matchback(k_height, k_width,outputH,outputW,padding,stride,greyscale,N,H,W)
        X[:,c,:,:] = target
    return X














class Transform:
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Note: we are not going to be accumulating gradients (where in hw1 we did)
        In each forward and backward pass, the gradients will be replaced.
        Therefore, there is no need to call on zero_grad().
        This is functionally the same as hw1 given that there is a step along the optimizer in each call of forward, backward, step
        """
        pass

    def update(self, learning_rate, momentum_coeff):
        """
        Apply gradients to update the parameters
        """
        pass


class ReLU(Transform):
    """
    Implement this class
    """
    def __init__(self):
        Transform.__init__(self)

    def forward(self, x):
        # compare x with 0
        relu = np.maximum(0,x)
        self.relu = relu
        return relu

    def backward(self, grad_wrt_out):
        return (np.multiply((self.relu>0).astype(int),grad_wrt_out))



class Flatten(Transform):
    """
    Implement this class
    """
    def forward(self, x):
        """
        returns Flatten(x)
        """
        self.bs = x.shape[0]
        self.c = x.shape[1]
        self.h = x.shape[2]
        self.w = x.shape[3]
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        
        
        return x

    def backward(self, dloss):
        """
        dLoss is the gradients wrt the output of Flatten
        returns gradients wrt the input to Flatten
        """
        dloss = dloss.reshape(self.bs,self.c,self.h,self.w)
        return dloss


class Conv(Transform):
    """
    Implement this class - Convolution Layer
    """
    def __init__(self, input_shape, filter_shape, rand_seed=0):
        """
        input_shape is a tuple: (channels, height, width)
        filter_shape is a tuple: (num of filters, filter height, filter width)
        weights shape (number of filters, number of input channels, filter height, filter width)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (num of filters, 1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        self.channels = input_shape[0]
        self.inputheight = input_shape[1]
        self.inputwidth = input_shape[2]
        
        self.nfilter = filter_shape[0]
        self.filterheight = filter_shape[1]
        self.filterwidth = filter_shape[2]


        self.b = np.sqrt(6/((self.nfilter+self.channels)*self.filterheight*self.filterwidth))
        # determine the size of the weight
        self.r = np.random.uniform(-self.b,self.b,self.nfilter*self.channels*self.filterheight*self.filterwidth)
        self.weights = self.r.reshape((self.nfilter,self.channels,self.filterheight,self.filterwidth))
        self.bias = np.zeros((self.nfilter,1))
        self.wk = np.zeros((self.nfilter,self.channels,self.filterheight,self.filterwidth))
        self.wb = np.zeros((self.nfilter,1))
  




    def forward(self, inputs, stride=1, pad=2):
        """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height, width)
        we recommend you use im2col here
        """
        self.inputs = inputs
        self.batchsize = inputs.shape[0]
        self.pad = pad
        self.stride = stride
        transinput =  im2col_helper.im2col(inputs, self.filterheight, self.filterwidth, self.pad, self.stride)
        filtermatrix = np.zeros((1,self.filterheight*self.filterwidth*self.channels))
        # stack all the filters
        self.weights = self.get_wb_conv()[0]
        for i in range (self.nfilter):
            kernel = self.weights[i]
            # flatten
            kernel = kernel.reshape(1,kernel.shape[0]*kernel.shape[1]*kernel.shape[2])
            filtermatrix = np.vstack((filtermatrix,kernel))
        # remove the dummy first row
        self.filtermatrix = filtermatrix[1:,:]




        fres = np.matmul(self.filtermatrix, transinput)
        # add bias term
        self.bias = self.get_wb_conv()[1]
        fres += self.bias
        
        self.outputH = (inputs.shape[2] - self.filterheight + 2*pad)//stride+1
        self.outputW = (inputs.shape[3] - self.filterwidth + 2*pad)//stride+1
        self.N = inputs.shape[0]



       
        result = np.transpose(fres.reshape(self.nfilter,self.outputH,self.outputW,self.N),(3,0,1,2))
        return result



    def backward(self, dloss):

        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        # reshape dloss
        self.dloss = dloss

        dout = np.transpose(dloss,(1,2,3,0))
        dout = dout.reshape(self.nfilter, self.outputH*self.outputW*self.N)

        bresx = np.matmul(self.filtermatrix.T,dout)
        dldx = im2col_helper.im2col_bw(bresx, self.inputs.shape, self.filterheight, self.filterwidth, self.pad, self.stride)
        
        #find dldk
        xcol = im2col_helper.im2col(self.inputs, self.filterheight, self.filterwidth, self.pad, self.stride)

        bresk = np.matmul(dout,xcol.T)
        dldk = bresk.reshape((self.nfilter,self.channels, self.filterheight, self.filterwidth))
        
        # find dldb

        dldb = np.sum(dout,axis =1)
        dldb = dldb.reshape((dldb.shape[0],1))
        self.dldk = dldk
        self.dldb = dldb
        self.dldx = dldx

        return ([dldk,dldb,dldx])

        
        
        


    def update(self, learning_rate=0.01, momentum_coeff=0.5):
        """
        Update weights and biases with gradients calculated by backward()
        Here we divide gradients by batch_size.
        """



        # calculate the gradient with momentum
        self.wk = momentum_coeff*self.wk - self.dldk/self.batchsize
        self.wb = momentum_coeff*self.wb - self.dldb/self.batchsize 

        self.weights += learning_rate*(self.wk)
        self.bias += learning_rate*(self.wb)
        

    def get_wb_conv(self):
        """
        Return weights and biases
        """
        return (self.weights,self.bias)


class MaxPool(Transform):
    """
    Implement this class - MaxPool layer
    """
    def __init__(self, filter_shape, stride):
        """
        filter_shape is (filter_height, filter_width)
        stride is a scalar
        """
        self.filterH = filter_shape[0]
        self.filterW = filter_shape[1]
        self.stride = stride

    def forward(self, inputs):
        self.inputs = inputs
        trans = im2col_helper.im2col(inputs, self.filterH, self.filterW, 0, self.stride)
        self.m = inputs.shape[0]
        self.channels = inputs.shape[1]

        self.H = inputs.shape[2]
        self.W = inputs.shape[3]
        # calculate the output width and the output height
        outputH = (self.H - self.filterH + 2*0)//self.stride+1
        outputW = (self.W - self.filterW + 2*0)//self.stride+1
        self.outputH = outputH
        self.outputW = outputW
        self.N = self.m
        # divid by channels and then find the max
        trans = trans.reshape(self.channels, trans.shape[0]//self.channels,-1)

        mpool = np.max(trans, axis = 1)
        self.mpoolindex = np.argmax(trans, axis = 1)

        mpool = np.transpose(mpool.reshape((self.channels, outputH, outputW,self.m)),(3,0,1,2))
 
        return mpool




    def backward(self, dloss):
        """
        dloss is the gradients wrt the output of forward()
        """
        dout = np.transpose(dloss,(1,2,3,0))
        dout = dout.reshape(self.channels, self.outputH*self.outputW*self.N)
        trans = im2col_helper.im2col(self.inputs, self.filterH, self.filterW, 0, self.stride)
        prereshape = trans.shape
        trans = trans.reshape(self.channels, trans.shape[0]//self.channels,-1)*0

        for i in range (trans.shape[0]):
            for j in range (trans.shape[2]):
                index = self.mpoolindex[i,j]

                trans[i,index,j] = dout[i,j]
        trans.reshape(prereshape)
        res = im2col_helper.im2col_bw(trans, self.inputs.shape, self.filterH, self.filterW, 0, self.stride)
        return res

        

                


        

  


        
    
     




class LinearLayer(Transform):
    """
    Implement this class - Linear layer
    """
    def __init__(self, indim, outdim, rand_seed=0):
        """
        indim, outdim: input and output dimensions
        weights shape (indim,outdim)
        Use Xavier initialization for weights, as instructed on handout
        Initialze biases as an array of zeros in shape of (outdim,1)
        """
        np.random.seed(rand_seed) # keep this line for autograding; you may remove it for training
        self.indim = indim
        self.outdim = outdim
        self.b = np.sqrt(6/(self.indim+self.outdim))
        # determine the size of the weight
        self.r = np.random.uniform(-self.b,self.b,self.indim*self.outdim)
        self.weights = self.r.reshape(self.indim,self.outdim)
        self.bias = np.zeros((self.outdim,1))
        self.wk = np.zeros((self.indim,self.outdim))
        self.wb = np.zeros((self.outdim,1))


    def forward(self, inputs):
        """
        Forward pass of linear layer
        inputs shape (batch_size, indim)
        """
        # Cited part of the part from my own HW1
        self.inputs = inputs
        self.indim = inputs.shape[0]
        self.input = inputs
        
        tw, tb = self.get_wb_fc()

        # matrix multiplication
        res = (np.matmul(inputs,tw))
        res += tb.T
        # store input
   

        return res

    def backward(self, dloss):
        """
        Read Transform.backward()'s docstring in this file
        dloss shape (batch_size, outdim)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
        """
        # Cited part of the part from my own HW1
        self.batchsize = dloss.shape[0]
        self.outdim = dloss.shape[1]
        
        tw, tb = self.get_wb_fc()
        dldh = np.matmul(dloss,self.weights.T)
        xtrans = np.transpose(self.inputs)

        # find out gradient wrt w
        self.gw = np.matmul(xtrans,dloss)

        self.gb = np.sum(dloss.T, axis=1)
        self.gb = self.gb.reshape(self.gb.shape[0],1)

        return [self.gw,self.gb,dldh]

    def update(self, learning_rate=0.01, momentum_coeff=0.5):
        """
        Similar to Conv.update()
        """
        # calculate the gradient with momentum
        self.wk = momentum_coeff*self.wk - self.gw/self.batchsize
        self.wb = momentum_coeff*self.wb - self.gb/self.batchsize 

        self.weights += learning_rate*(self.wk)
        self.bias += learning_rate*(self.wb)
        




    def get_wb_fc(self):
        """
        Return weights and biases as a tuple
        """
        return ((self.weights,self.bias))


class SoftMaxCrossEntropyLoss():
    """
    Implement this class
    """
    def labels2onehot(self,labels):
        return np.array([[i==lab for i in range(20)] for lab in labels])

    def forward(self, logits, labels, get_predictions=False):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in  the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should just be a sum of a batch, don't use mean)
        """
        logits = logits.T
        labels = labels.T
        self.z = np.exp(logits)

        self.softres = self.z/np.sum(self.z,axis=0)

        self.y = labels
        

        self.batch = logits.shape[1]
        logpi = np.log(self.softres)
        
        self.nyhat = np.argmax(self.softres,axis = 0)
        loss = np.sum(-(np.multiply(labels,logpi)))

        if (get_predictions==False):
            return loss
        else:
           
            yhat = self.labels2onehot(self.nyhat)

            return (loss, yhat)



    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't divide by batch_size here in order to pass autograding)
        """
        return (self.softres - self.y).T

    def getAccu(self):
        """
        Implement as you wish, not autograded.
        """



        totalcurrect = 0
        # in each batch
        for i in range (self.softres.shape[0]):
            predict = self.nyhat[i]

            ylabel = self.y[i,:]
            
            
            if (ylabel[predict] == True):
                totalcurrect+=1
        return totalcurrect


class ConvNet:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self,nfilter = 1):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 20 neurons
        Initialize SotMaxCrossEntropy object
        """
        Transform.__init__(self)
        self.conv = Conv((3,32,32),(nfilter,5,5))
        self.activation = ReLU()
        self.maxpool = MaxPool((2,2),2)
        self.flatten = Flatten()
        outputH = (32 - 2 + 2*0)//2+1
        outputW = (32 - 2 + 2*0)//2+1
        self.linear = LinearLayer (nfilter*outputH*outputW,20)
        self.softmax = SoftMaxCrossEntropyLoss()
        


    def forward(self, inputs, y_labels):

        inputs = self.conv.forward(inputs)
        inputs = self.activation.forward(inputs)
        inputs = self.maxpool.forward(inputs)


        inputs = self.flatten.forward(inputs)
        inputs = self.linear.forward(inputs)

        loss,inputs = self.softmax.forward(inputs,y_labels,True)
        return (loss,inputs) 
        

        
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """
        


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        grad = self.softmax.backward()
        grad = self.linear.backward(grad)[2]
        grad = self.flatten.backward(grad)

        grad = self.maxpool.backward(grad)
        grad = self.activation.backward(grad)
        grad = self.conv.backward(grad)[2]


       

    def update(self, learning_rate =  0.01, momentum_coeff = 0.5):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.linear.update(learning_rate, momentum_coeff)
        self.conv.update(learning_rate, momentum_coeff)
        


class ConvNetTwo:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool ->Conv -> Relu -> MaxPool -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self,nfilter = 1):
        """
        Initialize Conv, ReLU, MaxPool, Conv, ReLU,LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then Conv with filter size of 1x5x5
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 20 neurons
        Initialize SotMaxCrossEntropy object
        
        
        """
        Transform.__init__(self)

        self.conv1 = Conv((3,32,32),(nfilter,5,5))
        self.activation1 = ReLU()
        self.maxpool1 = MaxPool((2,2),2)

        self.conv2 = Conv((1,16,16),(nfilter,5,5))
        self.activation2 = ReLU()
        self.maxpool2 = MaxPool((2,2),2)

        self.flatten = Flatten()
        outputH = (16 - 2 + 2*0)//2+1
        outputW = (16 - 2 + 2*0)//2+1
        self.linear = LinearLayer (nfilter*outputH*outputW,20)
        self.softmax = SoftMaxCrossEntropyLoss()


    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """

        inputs = self.conv1.forward(inputs)
        inputs = self.activation1.forward(inputs)
        inputs = self.maxpool1.forward(inputs)


        inputs = self.conv2.forward(inputs)
        inputs = self.activation2.forward(inputs)
        inputs = self.maxpool2.forward(inputs)


        inputs = self.flatten.forward(inputs)
        inputs = self.linear.forward(inputs)

        loss,inputs = self.softmax.forward(inputs,y_labels,True)
        return (loss,inputs) 


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        grad = self.softmax.backward()
        grad = self.linear.backward(grad)[2]
        grad = self.flatten.backward(grad)

        grad = self.maxpool2.backward(grad)
        grad = self.activation2.backward(grad)
        grad = self.conv2.backward(grad)[2]

        grad = self.maxpool1.backward(grad)
        grad = self.activation1.backward(grad)
        grad = self.conv1.backward(grad)[2]
       

    def update(self, learning_rate =  0.01, momentum_coeff = 0.5):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.linear.update(learning_rate, momentum_coeff)
        self.conv2.update(learning_rate, momentum_coeff)
        self.conv1.update(learning_rate, momentum_coeff)

class ConvNetThree:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool ->Conv -> Relu -> MaxPool -> Conv -> Relu -> MaxPool ->Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self,nfilter=7):
        """
        Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
        Conv of input shape 3x32x32 with filter size of 7x3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        Conv of input shape 3x32x32 with filter size of 7x3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        Conv of input shape 3x32x32 with filter size of 7x3x3
        then apply Relu
        then perform MaxPooling with a 2x2 filter of stride 2
        then initialize linear layer with output 20 neurons
        Initialize SotMaxCrossEntropy object
        """
        Transform.__init__(self)
        nfilter = 7
        self.conv1 = Conv((3,32,32),(nfilter,3,3))
        self.activation1 = ReLU()
        self.maxpool1 = MaxPool((2,2),2)

        self.conv2 = Conv((7,17,17),(nfilter,3,3))
        self.activation2 = ReLU()
        self.maxpool2 = MaxPool((2,2),2)

        self.conv3 = Conv((7,9,9),(nfilter,3,3))
        self.activation3 = ReLU()
        self.maxpool3 = MaxPool((2,2),2)

        self.flatten = Flatten()
        outputH = (9 - 2 + 2*0)//2+1
        outputW = (9 - 2 + 2*0)//2+1
        self.linear = LinearLayer (nfilter*5*5,20)
        self.softmax = SoftMaxCrossEntropyLoss()



    def forward(self, inputs, y_labels):
        """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
        """


        inputs = self.conv1.forward(inputs)
        inputs = self.activation1.forward(inputs)
        inputs = self.maxpool1.forward(inputs)

        


        inputs = self.conv2.forward(inputs)
        inputs = self.activation2.forward(inputs)
        inputs = self.maxpool2.forward(inputs)

        

        inputs = self.conv3.forward(inputs)
        inputs = self.activation3.forward(inputs)
        inputs = self.maxpool3.forward(inputs)





        inputs = self.flatten.forward(inputs)
        inputs = self.linear.forward(inputs)


        loss,inputs = self.softmax.forward(inputs,y_labels,True)
        return (loss,inputs) 


    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        grad = self.softmax.backward()
        grad = self.linear.backward(grad)[2]
        grad = self.flatten.backward(grad)

        grad = self.maxpool3.backward(grad)
        grad = self.activation3.backward(grad)
        grad = self.conv3.backward(grad)[2]

        grad = self.maxpool2.backward(grad)
        grad = self.activation2.backward(grad)
        grad = self.conv2.backward(grad)[2]

        grad = self.maxpool1.backward(grad)
        grad = self.activation1.backward(grad)
        grad = self.conv1.backward(grad)[2]
       

    def update(self, learning_rate =  0.01, momentum_coeff = 0.5):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.linear.update(learning_rate, momentum_coeff)
        self.conv3.update(learning_rate, momentum_coeff)
        self.conv2.update(learning_rate, momentum_coeff)
        self.conv1.update(learning_rate, momentum_coeff)



# Implement the training as you wish. This part will not be autograded.
#Note: make sure to download the data from the provided link on page 17
if __name__ == '__main__':
    # This part may be helpful to write the training loop

    from argparse import ArgumentParser
    
    '''
    # Training parameters
    parser = ArgumentParser(description='CNN')
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--learning_rate', type=float, default = 0.01)
    parser.add_argument('--momentum', type=float, default = 0.5)
    parser.add_argument('--num_epochs', type=int, default = 50)
    parser.add_argument('--conv_layers', type=int, default = 1)
    parser.add_argument('--filters', type=int, default = 1)
    parser.add_argument('--title', type=str, default=None)
    args = parser.parse_args()
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    with open("data.pkl", "rb") as f:
        dict = pickle.load(f)
        train, test = dict["train"], dict["test"]
    f.close()

    train_data = train['data']
    test_data = test['data']
    train_labels = train['labels']
    test_labels = test['labels']

    #note: you should one-hot encode the labels

    num_train = len(train_data)
    num_test = len(test_data)
    batch_size = args.batch_size
    train_iter = num_train//batch_size + 1
    test_iter = num_test//batch_size + 1
    '''
    from matplotlib import pyplot as plt
    
    with open("data.pkl", "rb") as f:
        dict = pickle.load(f)
        train, test = dict["train"], dict["test"]
    f.close()
    def labels2onehot(labels):
        return np.array([[i==lab for i in range(20)] for lab in labels])
    trainX = train['data']
    testX = test['data']
    train_labels = train['labels']
    test_labels = test['labels']


    trainY = labels2onehot(train_labels)
    testY = labels2onehot(test_labels)

    # used some code in my own HW1
    atrainloss = []
    atrainaccuracy = []
    atestloss = []
    atestaccuracy = []
    def train(trainX,trainY,testX,testY,epoch,totalcases,batch,trainloss,testloss,trainaccu,testaccu,nfilter = 1):
        
        conv = ConvNetThree(nfilter)
     
        for i in range (0,epoch):
            
            ordering = np.random.permutation(8000)



            total_train_loss = 0
            

            totalTrainAcc = 0
            totalTestAcc = 0
            for j in range(0,totalcases,batch):
                print(i,j)
                

                # get the batch
                
      
                input = trainX[ordering[j:j+batch],:,:,:]
                #input = input.T
                
                # get the yhat
                # gety
                label = trainY[ordering[j:j+batch],:]
                #label = label.T
                # find train loss
                (tloss, yhat) = conv.forward(input,label)



                total_train_loss+= tloss

                
                totalTrainAcc += np.sum(np.all(yhat == label, axis=1))

                

                conv.backward()
                conv.update()

                
                

                
            # accu: (1- totalmiss/totalcases)
            # after iternation
            trainloss.append(total_train_loss /totalcases)
           
            trainaccu.append(totalTrainAcc/totalcases)


            totaltestloss = 0


            for j in range(0,500,batch):
                
                # get the batch
                input = testX[j:j+batch,:,:,:]
                #input = input.T
                
                # get the yhat
                # set train = False in testing
                
                
                # gety
                label = testY[j:j+batch,:]
                #label = label.T
                # find train loss
                (tloss, yhat) = conv.forward(input,label)


                totaltestloss+= tloss
                totalTestAcc += np.sum(np.all(yhat == label, axis=1))

                

                
                
            testloss.append(totaltestloss/500)
            testaccu.append(totalTestAcc/500)


            
            print(i)
    epoch =100
    train(trainX,trainY,testX,testY,epoch,8000,32,atrainloss,atestloss,atrainaccuracy,atestaccuracy,nfilter = 1)
    xlabel = list(range(0,epoch))
    # plot loss
    print(xlabel)
    print(atrainloss)
    print(atestloss)
    print(atrainaccuracy)
    print(atestaccuracy)
    plt.plot(np.array(xlabel),np.array(atrainloss),label = '3c7ftrain')
    plt.xticks(np.arange(min(xlabel), max(xlabel)+1, 10))
    plt.plot(np.array(xlabel),np.array(atestloss),label = '3c7ftest')
    plt.xticks(np.arange(min(xlabel), max(xlabel)+1, 10))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    leg = plt.legend(loc='upper right')

    plt.show()

    # plot accu
    plt.plot(xlabel,atrainaccuracy,label = '3c7ftrain')
    plt.xticks(np.arange(min(xlabel), max(xlabel)+1, 10))
    plt.plot(xlabel,atestaccuracy,label = '3c7ftest')
    plt.xticks(np.arange(min(xlabel), max(xlabel)+1, 10))
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    leg = plt.legend(loc='upper left')

    plt.show()



