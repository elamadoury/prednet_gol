import six
import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L


class ConvAE_mini(chainer.Chain):
    def __init__(self, input_size, input_size2, n_filters=10, n_latent=20, filter_size=3, activation='relu'):
        self.activation = {'relu': F.relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh}[activation]
        self.n_filters = n_filters
        self.n_latent = n_latent
        self.filter_size = filter_size
        self.dim1 = input_size - filter_size + 1
        self.dim2 = input_size2 - filter_size + 1
        super(ConvAE_mini, self).__init__()
        with self.init_scope():
            # encoder
            self.conv1 = L.Convolution2D(1, n_filters, filter_size, pad=1)
            self.lenc1 = L.Linear(n_filters*self.dim1*self.dim2, n_latent)
            # decoder
            self.ldec1 = L.Linear(n_latent, n_filters*self.dim1*self.dim2)
            self.deconv1 = L.Deconvolution2D(n_filters, 1, filter_size, pad=1)
            
    def forward(self, x, sigmoid=True):
        return self.decode(self.encode(x), x.data.shape[0], sigmoid)

    def encode(self, x):
        h1 = self.activation(self.conv1(x))
        # z = self.lenc1(h1)
        return h1
    
    def decode(self, z, batch_size, sigmoid=True):
        # h2 = F.reshape(self.activation(self.ldec1(z)), (batch_size, self.n_filters, self.dim1, self.dim1))
        h3 = self.deconv1(z)
        if sigmoid:
            return F.sigmoid(h3)
        else:
            return h3

    def get_loss_func(self, beta=1.0, k=1):
        def lf(x, x_next):
            z = self.encode(x)
            batchsize = len(z.data)
            # reconstruction loss
            loss = 0
            for l in six.moves.range(k):
                loss += F.bernoulli_nll(x_next, self.decode(z, batchsize, sigmoid=False)) \
                    / (k * batchsize)                
            self.loss = loss
            chainer.report({'loss': self.loss}, observer=self)
            return self.loss
        return lf

class ConvAE_mini_RGB(chainer.Chain):
    def __init__(self, input_size, input_size2, n_filters=10, n_latent=20, filter_size=3, activation='relu'):
        self.activation = {'relu': F.relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh}[activation]
        self.n_filters = n_filters
        self.n_latent = n_latent
        self.filter_size = filter_size
        self.dim1 = input_size - filter_size + 1
        self.dim2 = input_size2 - filter_size + 1
        super(ConvAE_mini_RGB, self).__init__()
        with self.init_scope():
            # encoder
            self.conv1 = L.Convolution2D(3, n_filters, filter_size, pad=1)
            self.lenc1 = L.Linear(n_filters*self.dim1*self.dim2, n_latent)
            # decoder
            self.ldec1 = L.Linear(n_latent, n_filters*self.dim1*self.dim2)
            self.deconv1 = L.Deconvolution2D(n_filters, 3, filter_size, pad=1)
            
    def forward(self, x, sigmoid=True):
        return self.decode(self.encode(x), x.data.shape[0], sigmoid)

    def encode(self, x):
        h1 = self.activation(self.conv1(x))
        # z = self.lenc1(h1)
        return h1
    
    def decode(self, z, batch_size, sigmoid=True):
        # h2 = F.reshape(self.activation(self.ldec1(z)), (batch_size, self.n_filters, self.dim1, self.dim1))
        h3 = self.deconv1(z)
        if sigmoid:
            return F.sigmoid(h3)
        else:
            return h3

    def get_loss_func(self, beta=1.0, k=1):
        def lf(x, x_next):
            z = self.encode(x)
            batchsize = len(z.data)
            # reconstruction loss
            loss = 0
            for l in six.moves.range(k):
                loss += F.bernoulli_nll(x_next, self.decode(z, batchsize, sigmoid=False)) \
                    / (k * batchsize)                
            self.loss = loss
            chainer.report({'loss': self.loss}, observer=self)
            return self.loss
        return lf


