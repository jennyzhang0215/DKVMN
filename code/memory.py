import mxnet as mx

class DKVMNHeadGroup(object):
    def __init__(self, memory_size, memory_state_dim, is_write, name="DKVMNHeadGroup"):
        """"
        Parameters
            memory_size:        scalar
            memory_state_dim:   scalar
            is_write:           boolean
        """
        self.name = name
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        if self.is_write:
            self.erase_signal_weight = mx.sym.Variable(name=name + ":erase_signal_weight")
            self.erase_signal_bias = mx.sym.Variable(name=name + ":erase_signal_bias")
            self.add_signal_weight = mx.sym.Variable(name=name + ":add_signal_weight")
            self.add_signal_bias = mx.sym.Variable(name=name + ":add_signal_bias")


    def addressing(self, control_input, memory):
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim)
            memory:                 Shape (memory_size, memory_state_dim)
        Returns
            correlation_weight:     Shape (batch_size, memory_size)
        """
        similarity_score = mx.sym.FullyConnected(data=control_input,
                                                 num_hidden=self.memory_size,
                                                 weight=memory,
                                                 no_bias=True,
                                                 name="similarity_score")
        correlation_weight = mx.sym.SoftmaxActivation(similarity_score) # Shape: (batch_size, memory_size)
        return correlation_weight

    def read(self, memory, control_input=None, read_weight=None ):
        """
        Parameters
            control_input:  Shape (batch_size, control_state_dim)
            memory:         Shape (batch_size, memory_size, memory_state_dim)
            read_weight:    Shape (batch_size, memory_size)
        Returns
            read_content:   Shape (batch_size,  memory_state_dim)
        """
        if read_weight is None:
            read_weight = self.addressing(control_input=control_input, memory=memory)
        read_weight = mx.sym.Reshape(read_weight, shape=(-1,1,self.memory_size))
        read_content = mx.sym.Reshape(data=mx.sym.batch_dot(read_weight, memory), # Shape (batch_size, 1, memory_state_dim)
                                 shape=(-1,self.memory_state_dim)) # Shape (batch_size, memory_state_dim)
        return read_content


    def write(self, control_input, memory, write_weight=None):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """
        assert self.is_write
        if write_weight is None:
            write_weight = self.addressing(control_input=control_input, memory=memory)  # Shape Shape (batch_size, memory_size)
        ## erase_signal  Shape (batch_size, memory_state_dim)
        erase_signal = mx.sym.FullyConnected(data=control_input,
                                             num_hidden=self.memory_state_dim,
                                             weight=self.erase_signal_weight,
                                             bias=self.erase_signal_bias)
        erase_signal = mx.sym.Activation(data=erase_signal, act_type='sigmoid', name=self.name + "_erase_signal")
        ## add_signal  Shape (batch_size, memory_state_dim)
        add_signal = mx.sym.FullyConnected(data=control_input,
                                           num_hidden=self.memory_state_dim,
                                           weight=self.add_signal_weight,
                                           bias=self.add_signal_bias)
        add_signal = mx.sym.Activation(data=add_signal, act_type='tanh', name=self.name + "_add_signal")
        ## erase_mult  Shape (batch_size, memory_size, memory_state_dim)
        erase_mult = 1 - mx.sym.batch_dot(mx.sym.Reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                          mx.sym.Reshape(erase_signal, shape=(-1, 1, self.memory_state_dim)))

        aggre_add_signal = mx.sym.batch_dot(mx.sym.Reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                          mx.sym.Reshape(add_signal, shape=(-1, 1, self.memory_state_dim)))
        new_memory = memory * erase_mult + aggre_add_signal
        return new_memory

class DKVMN(object):
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim,
                 init_memory_key=None, init_memory_value=None, name="DKVMN"):
        """
        :param memory_size:             scalar
        :param memory_key_state_dim:    scalar
        :param memory_value_state_dim:  scalar
        :param init_memory_key:         Shape (memory_size, memory_value_state_dim)
        :param init_memory_value:       Shape (batch_size, memory_size, memory_value_state_dim)
        """
        self.name = name
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.init_memory_key = mx.sym.Variable(self.name + ":init_memory_key_weight") if init_memory_key is None\
                                                                               else init_memory_key
        self.init_memory_value = mx.sym.Variable(self.name + ":init_memory_value_weight") if init_memory_value is None\
                                                                               else init_memory_value
        self.key_head = DKVMNHeadGroup(memory_size = self.memory_size,
                                       memory_state_dim = self.memory_key_state_dim,
                                       is_write = False,
                                       name = self.name + "->key_head")

        self.value_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                         memory_state_dim=self.memory_value_state_dim,
                                         is_write=True,
                                         name=self.name + "->value_head")
        self.memory_key = self.init_memory_key
        self.memory_value = self.init_memory_value

    def attention(self, control_input):
        assert isinstance(control_input, mx.symbol.Symbol)
        correlation_weight = self.key_head.addressing(control_input=control_input, memory=self.memory_key)
        return correlation_weight #(batch_size, memory_size)

    def read(self, read_weight):
        read_content = self.value_head.read(memory=self.memory_value, read_weight=read_weight)
        return read_content  #(batch_size, memory_state_dim)


    def write(self, write_weight, control_input):
        assert isinstance(control_input, mx.symbol.Symbol)
        self.memory_value = self.value_head.write(control_input=control_input,
                                                  memory=self.memory_value,
                                                  write_weight=write_weight)
        return self.memory_value
