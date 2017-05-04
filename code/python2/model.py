import mxnet as mx
import mxnet.ndarray as nd
import ast as ast
from memory import DKVMN

def safe_eval(expr):
    if type(expr) is str:
        return ast.literal_eval(expr)
    else:
        return expr

class LogisticRegressionMaskOutput(mx.operator.CustomOp):
    def __init__(self, ignore_label):
        super(LogisticRegressionMaskOutput, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], 1.0 / (1.0 + nd.exp(- in_data[0])))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        output = out_data[0].asnumpy()
        label = in_data[1].asnumpy()
        data_grad = (output - label) * (label != self.ignore_label)
        self.assign(in_grad[0], req[0], data_grad)

@mx.operator.register("LogisticRegressionMaskOutput")
class LogisticRegressionMaskOutputProp(mx.operator.CustomOpProp):
    def __init__(self, ignore_label):
        super(LogisticRegressionMaskOutputProp, self).__init__(need_top_grad=False)
        self.ignore_label = safe_eval(ignore_label)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return LogisticRegressionMaskOutput(ignore_label=self.ignore_label)

def logistic_regression_mask_output(data, label, ignore_label, name=None):
    return mx.sym.Custom(name=name,
                         op_type="LogisticRegressionMaskOutput",
                         ignore_label=ignore_label,
                         data=data,
                         label=label)

class MODEL(object):
    def __init__(self, n_question, seqlen, batch_size,
                 q_embed_dim, qa_embed_dim,
                 memory_size, memory_key_state_dim, memory_value_state_dim,
                 # lambda1, lambda2,
                 final_fc_dim, name="KT"):
        self.n_question = n_question
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.final_fc_dim = final_fc_dim
        self.name = name

    def sym_gen(self):
        ### TODO input variable 'q_data'
        q_data = mx.sym.Variable('q_data', shape=(self.seqlen, self.batch_size)) # (seqlen, batch_size)
        ### TODO input variable 'qa_data'
        qa_data = mx.sym.Variable('qa_data', shape=(self.seqlen, self.batch_size))  # (seqlen, batch_size)
        ### TODO input variable 'target'
        target = mx.sym.Variable('target', shape=(self.seqlen, self.batch_size)) #(seqlen, batch_size)

        ### Initialize Memory
        init_memory_key = mx.sym.Variable('init_memory_key_weight')
        init_memory_value = mx.sym.Variable('init_memory_value',
                                            shape=(self.memory_size, self.memory_value_state_dim),
                                            init=mx.init.Normal(0.1)) # (self.memory_size, self.memory_value_state_dim)
        init_memory_value = mx.sym.broadcast_to(mx.sym.expand_dims(init_memory_value, axis=0),
                                                shape=(self.batch_size, self.memory_size, self.memory_value_state_dim))

        mem = DKVMN(memory_size=self.memory_size,
                   memory_key_state_dim=self.memory_key_state_dim,
                   memory_value_state_dim=self.memory_value_state_dim,
                   init_memory_key=init_memory_key,
                   init_memory_value=init_memory_value,
                   name="DKVMN")


        ### embedding
        q_data = mx.sym.BlockGrad(q_data)
        q_embed_data = mx.sym.Embedding(data=q_data, input_dim=self.n_question+1,
                                        output_dim=self.q_embed_dim, name='q_embed')
        slice_q_embed_data = mx.sym.SliceChannel(q_embed_data, num_outputs=self.seqlen, axis=0, squeeze_axis=True)

        qa_data = mx.sym.BlockGrad(qa_data)
        qa_embed_data = mx.sym.Embedding(data=qa_data, input_dim=self.n_question*2+1,
                                         output_dim=self.qa_embed_dim, name='qa_embed')
        slice_qa_embed_data = mx.sym.SliceChannel(qa_embed_data, num_outputs=self.seqlen, axis=0, squeeze_axis=True)

        value_read_content_l = []
        input_embed_l = []
        for i in range(self.seqlen):
            ## Attention
            q = slice_q_embed_data[i]
            correlation_weight = mem.attention(q)

            ## Read Process
            read_content = mem.read(correlation_weight) #Shape (batch_size, memory_state_dim)
            ### save intermedium data
            value_read_content_l.append(read_content)
            input_embed_l.append(q)

            ## Write Process
            qa = slice_qa_embed_data[i]
            new_memory_value = mem.write(correlation_weight, qa)

        all_read_value_content = mx.sym.Concat(*value_read_content_l, num_args=self.seqlen, dim=0)

        input_embed_content = mx.sym.Concat(*input_embed_l, num_args=self.seqlen, dim=0)
        input_embed_content = mx.sym.FullyConnected(data=input_embed_content, num_hidden=50, name="input_embed_content")
        input_embed_content = mx.sym.Activation(data=input_embed_content, act_type='tanh', name="input_embed_content_tanh")

        read_content_embed = mx.sym.FullyConnected(data=mx.sym.Concat(all_read_value_content, input_embed_content, num_args=2, dim=1),
                                                   num_hidden=self.final_fc_dim, name="read_content_embed")
        read_content_embed = mx.sym.Activation(data=read_content_embed, act_type='tanh', name="read_content_embed_tanh")

        pred = mx.sym.FullyConnected(data=read_content_embed, num_hidden=1, name="final_fc")

        pred_prob = logistic_regression_mask_output(data=mx.sym.Reshape(pred, shape=(-1, )),
                                                    label=mx.sym.Reshape(data=target, shape=(-1,)),
                                                    ignore_label=-1., name='final_pred')
        return mx.sym.Group([pred_prob])
