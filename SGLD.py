import keras
from keras import backend as K
from keras.optimizers import SGD

class SGLD(keras.optimizers.SGD):
    def __init__(self, inv_temp=0.01, **kwargs):
        super(SGLD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.inv_temp = K.variable(inv_temp, name='inv_temp')

    def get_gradients(self, loss, params):
            grads = super(SGLD, self).get_gradients(loss, params)

            # Add decayed gaussian noise
            # t = K.cast(self.iterations, K.dtype(grads[0]))
            variance = 2. / self.lr / self.inv_temp
            grads = [
                grad + K.random_normal(
                    grad.shape,
                    mean=0.0,
                    stddev=K.sqrt(variance),
                    dtype=K.dtype(grads[0])
                )
                for grad in grads
            ]

            return grads

    def get_config(self):
        config = {'inv_temp': float(K.get_value(self.inv_temp))}
        base_config = super(SGLD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))