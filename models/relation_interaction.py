import keras.backend as K
from keras.engine import Model

from layers.SimilarityMatrix import SimilarityMatrix
from lib.Evaluator import Evaluator
from models.relation_attention import output_shape2
from models.relation_base_model import BaseModel
from keras.layers import Concatenate, GRU, TimeDistributed, Dense, Dropout, Bidirectional, Lambda, Input, Dot, Subtract, \
    Multiply, Permute, GlobalAvgPool1D, GlobalMaxPool1D
import numpy as np
from keras.activations import softmax


MAX_LEN = 125


def reduce_dimension(x, length, mask):
    res = K.reshape(x, [-1, length])  # (?, 78)
    res = K.softmax(res)
    res = res * K.cast(mask, dtype='float32')  # (?, 78)
    temp = K.sum(res, axis=1, keepdims=True)  # (?, 1)
    temp = K.repeat_elements(temp, rep=length, axis=1)  # (?, 78)
    return res / temp


def reduce_dimension_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    return [shape[0], shape[1]]


def output_shape(input_shape):
    return [input_shape[0], input_shape[-1]]


def average(x):
    res = K.sum(x, axis=1, keepdims=True)  # (?, 1)
    res = x / res  # (?, 125)
    return res


def no_change(input_shape):
    return input_shape


def liter(x, length):
    res = K.repeat(x, length)  # (?, 82, 300)
    return res


def liter_output_shape(input_shape):
    shape = list(input_shape)
    return [shape[0], MAX_LEN, shape[1]]


def attention(x, dim):
    res = K.batch_dot(x[0], x[1], axes=[1, 1])
    return K.reshape(res, [-1, dim])


def attention_output_shape(input_shape):
    shape = list(input_shape[1])
    assert len(shape) == 3
    return [shape[0], shape[2]]


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = Subtract()([input_1, input_2])
    out_= Concatenate()([sub, mult])
    return out_


def soft_attention_alignment(input_1, input_2):
    """Align text representation with neural soft attention"""

    attention = Dot(axes=-1)([input_1, input_2])

    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=no_change)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                              output_shape=no_change)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def minus_soft_attention_alignment(input_1, input_2):
    """Align text representation with neural soft attention"""

    attention = SimilarityMatrix()([input_1, input_2])

    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=no_change)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                              output_shape=no_change)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = Subtract()([input_1, input_2])
    out_= Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


class RI(BaseModel):

    def __init__(self, max_len=125, class_num=9, use_development_set=True):

        super(RI, self).__init__(use_development_set)

        self.max_len = max_len
        self.class_num = class_num
        self.name = "interaction_model"

    def build_model(self):

        sentence_t = Concatenate()([self.sen_embedding,
                                    self.sen_entity_type_embedding,
                                    self.position_t_embedding,])

        sentence_e = Concatenate()([self.sen_embedding,
                                    self.sen_entity_type_embedding,
                                    self.position_a_embedding,])

        encoding_layer = Bidirectional(GRU(300,
                                           activation="relu",
                                           return_sequences=True,
                                           recurrent_dropout=0.3,
                                           dropout=0.3))

        sentence_t = encoding_layer(sentence_t)
        sentence_e = encoding_layer(sentence_e)

        st_aligned, se_aligned = minus_soft_attention_alignment(sentence_t, sentence_e)
        sentence_t = Concatenate()([sentence_t, se_aligned, submult(sentence_t, se_aligned)])
        sentence_e = Concatenate()([sentence_e, st_aligned, submult(sentence_e, st_aligned)])

        encoding_layer_2 = Bidirectional(GRU(300,
                                             activation="relu",
                                             return_sequences=True,
                                             recurrent_dropout=0.3,
                                             dropout=0.3))

        sentence_t = encoding_layer_2(sentence_t)
        sentence_e = encoding_layer_2(sentence_e)

        rep_t = apply_multiple(sentence_t, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        rep_e = apply_multiple(sentence_e, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        # average_layer = Lambda(average, output_shape=no_change)
        # position_mt = average_layer(self.position_mt)
        # position_ma = average_layer(self.position_ma)

        # triggers = Lambda(liter,
        #                   output_shape=liter_output_shape,
        #                   arguments={'length': self.max_len})(trigger)  # (?, 125, 600)


        # sentence = Bidirectional(GRU(300,
        #                              activation="relu",
        #                              return_sequences=False,
        #                              recurrent_dropout=0.3,
        #                              dropout=0.3))(sentence)

        x_layer = Lambda(lambda x: K.reshape(x, [-1, self.TRIGGER_TYPE_VEC_DIM]), output_shape=output_shape)
        trigger_type = x_layer(self.trigger_type_embedding)
        entity_type = x_layer(self.entity_type_embedding)

        x = Concatenate()([trigger_type, entity_type, rep_t, rep_e])
        x = Dropout(rate=0.5)(x)
        output = Dense(9, activation='softmax')(x)

        return output

    def train_model(self, max_epoch=30):

        evaluator = Evaluator(true_labels=self.test_labels, sentences=self.test_sentence_words_input,
                              position_mt=self.test_position_mt, position_me=self.test_position_ma)
        log = open("../log/" + self.name + ".txt", 'a+', encoding='utf-8')
        for i in range(max_epoch):
            self.model.fit({'sentence_word': self.train_sentence_words_input,
                            'sentence_entity_type': self.train_sentence_entity_inputs,
                            'position_t': self.train_position_t,
                            'position_a': self.train_position_a,
                            'trigger_type': self.train_trigger_type,
                            'entity_type': self.train_entity_type,
                            'trigger_mask': self.train_position_mt,
                            'entity_mask': self.train_position_ma
                            },
                           self.train_labels,
                           epochs=1,
                           batch_size=256,
                           verbose=1)

            print("# -- test set --- #")
            results = self.model.predict({'sentence_word': self.test_sentence_words_input,
                                          'sentence_entity_type': self.test_sentence_entity_inputs,
                                          'position_t': self.test_position_t,
                                          'position_a': self.test_position_a,
                                          'trigger_type': self.test_trigger_type,
                                          'entity_type': self.test_entity_type,
                                          'trigger_mask': self.test_position_mt,
                                          'entity_mask': self.test_position_ma
                                         },
                                         batch_size=128,
                                         verbose=0)

            print("--------------epoch " + str(i + 1) + " ---------------------")
            macro_f1, micro_F1, p, r = evaluator.get_f1(predictions=results, epoch=i + 1)

            log.write("epoch: " + str(i + 1) + " " + str(p) + " " + str(r) + " " + str(micro_F1) + "\n")
            if (i + 1) % 5 == 0:
                print("current max macro_F1 score: " + str(evaluator.max_macro_F1 * 100))
                print("max macro_F1 is gained in epoch " + str(evaluator.max_macro_F1_epoch))
                print("current max micro_F1 score: " + str(evaluator.max_micro_F1 * 100))
                print("max micro_F1 is gained in epoch " + str(evaluator.max_micro_F1_epoch))

                log.write("current max macro_F1 score: " + str(evaluator.max_macro_F1 * 100) + "\n")
                log.write("max macro_F1 is gained in epoch " + str(evaluator.max_macro_F1_epoch) + "\n")
                log.write("current max micro_F1 score: " + str(evaluator.max_micro_F1 * 100) + "\n")
                log.write("max micro_F1 is gained in epoch " + str(evaluator.max_micro_F1_epoch) + "\n")
            print("------------------------------------------------------------")
        log.close()

    def make_input(self):

        inputs = [None] * 8

        inputs[0] = Input(shape=(self.max_len,), dtype='int32', name='sentence_word')
        inputs[1] = Input(shape=(self.max_len,), dtype='int32', name='sentence_entity_type')
        inputs[2] = Input(shape=(self.max_len,), dtype='int32', name='position_t')
        inputs[3] = Input(shape=(self.max_len,), dtype='int32', name='position_a')
        inputs[4] = Input(shape=(1,), dtype='int32', name='trigger_type')
        inputs[5] = Input(shape=(1,), dtype='int32', name='entity_type')

        inputs[6] = Input(shape=(self.max_len,), dtype='float32', name='trigger_mask')
        inputs[7] = Input(shape=(self.max_len,), dtype='float32', name='entity_mask')

        return inputs

    def compile_model(self):
        inputs = [self.sen_input, self.sen_entity_type_input, self.position_t,
                  self.position_a, self.trigger_type_input, self.entity_type_input,
                  self.position_mt, self.position_ma] = self.make_input()

        [self.sen_embedding, self.sen_entity_type_embedding, self.position_t_embedding, self.position_a_embedding,
         self.trigger_type_embedding, self.entity_type_embedding] = self.embedded()

        self.output = self.build_model()

        self.model = Model(inputs=inputs, outputs=self.output)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['acc'])
        print(self.model.summary())


if __name__ == '__main__':

    s = RI(max_len=125, class_num=9, use_development_set=False)
    for i in range(5):
        s.compile_model()
        s.train_model(max_epoch=35)
