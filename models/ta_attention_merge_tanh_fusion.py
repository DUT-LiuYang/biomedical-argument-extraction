

import keras.backend as K
from keras.engine import Model

from lib.Evaluator import Evaluator

from models.relation_base_model import BaseModel
from keras.layers import Concatenate, GRU, TimeDistributed, Dense, Dropout, Bidirectional, Lambda, Input, Dot, Embedding
import numpy as np

from models.relation_mask_attention import output_shape2

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


class Fusion_ta(BaseModel):

    def __init__(self, max_len=125, class_num=9, use_development_set=True):

        super(Fusion_ta, self).__init__(use_development_set)

        self.max_len = max_len
        self.class_num = class_num
        self.name = "ta_attention_merge_tanh_fusion"

    def build_model(self):
        sentence = Concatenate()([self.sen_embedding,
                                  # self.sen_entity_type_embedding,
                                  self.position_t_embedding,
                                  self.position_a_embedding])

        sentence = Bidirectional(GRU(300,
                                     activation="relu",
                                     return_sequences=True,
                                     recurrent_dropout=0.3,
                                     dropout=0.3))(sentence)

        average_layer = Lambda(average, output_shape=no_change)
        position_mt = average_layer(self.position_mt)
        position_ma = average_layer(self.position_ma)

        trigger = Dot(axes=[1, 1])([sentence, position_mt])
        entity = Dot(axes=[1, 1])([sentence, position_ma])

        triggers = Lambda(liter,
                          output_shape=liter_output_shape,
                          arguments={'length': self.max_len})(trigger)  # (?, 125, 300)
        entities = Lambda(liter,
                          output_shape=liter_output_shape,
                          arguments={'length': self.max_len})(entity)  # (?, 125, 300)

        x = Concatenate()([triggers, entities, sentence])  # (?, 125, 1200)
        x = Dense(300, activation='tanh')(x)   # (?, 82, 600)
        x = Dense(1)(x)  # (?, 125, 1)

        x = Lambda(reduce_dimension,
                   output_shape=reduce_dimension_output_shape,
                   arguments={'length': self.max_len},
                   mask=self.sentence_embedding_layer.get_output_mask_at(0),
                   name='aspect_attention')(x)  # (?, 125)
        x = Lambda(attention, output_shape=attention_output_shape, arguments={'dim': 600})([x, sentence])  # (?, 600)

        x = Dropout(rate=0.5)(x)
        output = Dense(9, activation='softmax')(x)

        return output

    def train_model(self, max_epoch=30):

        evaluator = Evaluator(true_labels=self.test_labels, sentences=self.test_sentence_words_input,
                              position_mt=self.test_position_mt, position_me=self.test_position_ma,
                              correction_factor=self.correction_factor, name=self.name)
        log = open("../log/" + self.name + ".txt", 'a+', encoding='utf-8')
        for i in range(max_epoch):
            self.model.fit({'sentence_word': self.train_sentence_words_input,
                            # 'sentence_entity_type': self.train_sentence_entity_inputs,
                            'position_t': self.train_position_t,
                            'position_a': self.train_position_a,
                            # 'trigger_type': self.train_trigger_type,
                            # 'entity_type': self.train_entity_type,
                            'trigger_mask': self.train_position_mt,
                            'entity_mask': self.train_position_ma
                            },
                           self.train_labels,
                           epochs=1,
                           batch_size=256,
                           verbose=1)

            print("# -- test set --- #")
            results = self.model.predict({'sentence_word': self.test_sentence_words_input,
                                          # 'sentence_entity_type': self.test_sentence_entity_inputs,
                                          'position_t': self.test_position_t,
                                          'position_a': self.test_position_a,
                                          # 'trigger_type': self.test_trigger_type,
                                          # 'entity_type': self.test_entity_type,
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

        inputs = [None] * 5

        inputs[0] = Input(shape=(self.max_len,), dtype='int32', name='sentence_word')

        inputs[1] = Input(shape=(self.max_len,), dtype='int32', name='position_t')
        inputs[2] = Input(shape=(self.max_len,), dtype='int32', name='position_a')

        inputs[3] = Input(shape=(self.max_len,), dtype='float32', name='trigger_mask')
        inputs[4] = Input(shape=(self.max_len,), dtype='float32', name='entity_mask')

        return inputs

    def compile_model(self):
        inputs = [self.sen_input, self.position_t,
                  self.position_a,
                  self.position_mt, self.position_ma] = self.make_input()

        [self.sen_embedding, self.position_t_embedding, self.position_a_embedding] = self.embedded()

        self.output = self.build_model()

        self.model = Model(inputs=inputs, outputs=self.output)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['acc'])
        print(self.model.summary())

    def embedded(self):

        self.sentence_embedding_layer = Embedding(self.num_words + 2,
                                                  self.EMBEDDING_DIM,
                                                  weights=[self.embedding_matrix],
                                                  input_length=self.max_len,
                                                  trainable=False,
                                                  mask_zero=True)
        sentence_embedding = self.sentence_embedding_layer(self.sen_input)

        position_t_embedding_layer = Embedding(125,
                                               self.POSITION_VEC_DIM,
                                               input_length=self.max_len,
                                               trainable=True,
                                               mask_zero=False)
        position_t_embedding = position_t_embedding_layer(self.position_t)

        position_a_embedding_layer = Embedding(125,
                                               self.POSITION_VEC_DIM,
                                               input_length=self.max_len,
                                               trainable=True,
                                               mask_zero=False)
        position_a_embedding = position_a_embedding_layer(self.position_a)

        return [sentence_embedding, position_t_embedding, position_a_embedding]


if __name__ == '__main__':

    s = Fusion_ta(max_len=125, class_num=9, use_development_set=False)
    for i in range(5):
        s.compile_model()
        s.train_model(max_epoch=35)
