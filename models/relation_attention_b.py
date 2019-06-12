import keras.backend as K
from keras.engine import Model

from lib.Evaluator import Evaluator
from models.realation_fusion import attention, attention_output_shape
from models.relation_base_model import BaseModel
from keras.layers import Concatenate, GRU, Dense, Dropout, Bidirectional, Lambda, Input, Embedding
import numpy as np

from models.relation_mask_attention import output_shape2


def output_shape(input_shape):
    return [input_shape[0], input_shape[-1]]


def reduce_dimension(x, length, mask):
    res = K.squeeze(x, axis=2)  # (?, 125)
    res = res - (1 - K.cast(mask, dtype='float32')) * 100
    res = K.softmax(res)
    return res
    # print(np.shape(mask))
    # temp = K.sum(res, axis=1, keepdims=True)  # (?, 1)
    # temp = K.repeat_elements(temp, rep=length, axis=1)  # (?, 125)
    # return res / temp


def reduce_dimension_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    return [shape[0], shape[1]]


class PureAttention(BaseModel):

    def __init__(self, max_len=125, class_num=9, use_development_set=True):

        super(PureAttention, self).__init__(use_development_set)

        self.max_len = max_len
        self.class_num = class_num
        self.name = "pure_attention_mask"

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
        # x = Dense(1, activation='tanh')(sentence)  # (?, 125, 1)
        x = Dense(1)(sentence)  # (?, 125, 1)
        x = Lambda(reduce_dimension,
                   output_shape=reduce_dimension_output_shape,
                   arguments={'length': self.max_len},
                   mask=self.sentence_embedding_layer.get_output_mask_at(0),
                   name='aspect_attention')(x)  # (?, 125)

        # attention_x = Lambda(lambda x: K.softmax(K.squeeze(x, axis=2)), output_shape=output_shape2)(x)  # (?, 125)

        x = Lambda(attention, output_shape=attention_output_shape, arguments={'dim': 600})([x, sentence])  # (?, 600)

        x = Dropout(rate=0.5)(x)
        output = Dense(9, activation='softmax')(x)

        return output

    def train_model(self, max_epoch=30):

        evaluator = Evaluator(true_labels=self.test_labels, sentences=self.test_sentence_words_input,
                              position_mt=self.test_position_mt, position_me=self.test_position_ma,
                              correction_factor=self.correction_factor)
        log = open("../log/" + self.name + ".txt", 'a+', encoding='utf-8')
        for i in range(max_epoch):
            self.model.fit({'sentence_word': self.train_sentence_words_input,
                            # 'sentence_entity_type': self.train_sentence_entity_inputs,
                            'position_t': self.train_position_t,
                            'position_a': self.train_position_a,
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

        inputs = [None] * 3

        inputs[0] = Input(shape=(self.max_len,), dtype='int32', name='sentence_word')
        inputs[1] = Input(shape=(self.max_len,), dtype='int32', name='position_t')
        inputs[2] = Input(shape=(self.max_len,), dtype='int32', name='position_a')

        return inputs

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

    def compile_model(self):
        inputs = [self.sen_input, self.position_t,
                  self.position_a] = self.make_input()

        [self.sen_embedding, self.position_t_embedding, self.position_a_embedding] = self.embedded()

        self.output = self.build_model()

        self.model = Model(inputs=inputs, outputs=self.output)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['acc'])
        print(self.model.summary())


if __name__ == '__main__':

    s = PureAttention(max_len=125, class_num=9, use_development_set=False)
    for i in range(4):
        s.compile_model()
        s.train_model(max_epoch=35)
