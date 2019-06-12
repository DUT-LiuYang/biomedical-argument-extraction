import keras.backend as K
from keras.engine import Model

from lib.Evaluator import Evaluator

from models.relation_base_model import BaseModel
from keras.layers import Concatenate, GRU, Dense, Dropout, Bidirectional, Lambda, Embedding, Input
import numpy as np


def output_shape(input_shape):
    return [input_shape[0], input_shape[-1]]


class Baseline(BaseModel):

    def __init__(self, max_len=125, class_num=9, use_development_set=True):

        super(Baseline, self).__init__(use_development_set)

        self.max_len = max_len
        self.class_num = class_num
        self.name = "baseline_b"

    def build_model(self):
        sentence = Concatenate()([self.sen_embedding,
                                  # self.sen_entity_type_embedding,
                                  self.position_t_embedding,
                                  self.position_a_embedding])

        sentence = Bidirectional(GRU(300,
                                     activation="relu",
                                     return_sequences=False,
                                     recurrent_dropout=0.3,
                                     dropout=0.3))(sentence)

        x_layer = Lambda(lambda x: K.reshape(x, [-1, self.TRIGGER_TYPE_VEC_DIM]), output_shape=output_shape)
        trigger_type = x_layer(self.trigger_type_embedding)
        entity_type = x_layer(self.entity_type_embedding)

        x = Concatenate()([sentence, trigger_type, entity_type])
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
                            'trigger_type': self.train_trigger_type,
                            'entity_type': self.train_entity_type
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
                                          'trigger_type': self.test_trigger_type,
                                          'entity_type': self.test_entity_type
                                         },
                                         batch_size=128,
                                         verbose=0)

            print("--------------epoch " + str(i + 1) + " ---------------------")
            macro_f1, micro_F1, p, r = evaluator.get_f1(predictions=results, epoch=i + 1)

            log.write("epoch-" + str(i + 1) + ": " + str(p * 100) + " " + str(r * 100) + " " + str(micro_F1 * 100) + "\n")

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

    def compile_model(self):
        inputs = [self.sen_input, self.position_t,
                  self.position_a, self.trigger_type_input, self.entity_type_input] = self.make_input()

        [self.sen_embedding, self.position_t_embedding, self.position_a_embedding,
         self.trigger_type_embedding, self.entity_type_embedding] = self.embedded()

        self.output = self.build_model()

        self.model = Model(inputs=inputs, outputs=self.output)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['acc'])
        print(self.model.summary())

    def make_input(self):

        inputs = [None] * 5

        inputs[0] = Input(shape=(self.max_len,), dtype='int32', name='sentence_word')
        # inputs[1] = Input(shape=(self.max_len,), dtype='int32', name='sentence_entity_type')
        inputs[1] = Input(shape=(self.max_len,), dtype='int32', name='position_t')
        inputs[2] = Input(shape=(self.max_len,), dtype='int32', name='position_a')
        inputs[3] = Input(shape=(1,), dtype='int32', name='trigger_type')
        inputs[4] = Input(shape=(1,), dtype='int32', name='entity_type')

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

        trigger_type_embedding_layer = Embedding(38,
                                                 self.TRIGGER_TYPE_VEC_DIM,
                                                 input_length=1,
                                                 trainable=True,
                                                 mask_zero=False)

        trigger_type_embedding = trigger_type_embedding_layer(self.trigger_type_input)
        entity_type_embedding = trigger_type_embedding_layer(self.entity_type_input)

        return [sentence_embedding, position_t_embedding, position_a_embedding,
                trigger_type_embedding, entity_type_embedding]


if __name__ == '__main__':

    s = Baseline(max_len=125, class_num=9, use_development_set=False)
    for i in range(3):
        s.compile_model()
        s.train_model(max_epoch=40)
