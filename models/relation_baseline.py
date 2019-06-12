import keras.backend as K

from lib.Evaluator import Evaluator
from models.relation_attention import RA
from models.relation_base_model import BaseModel
from keras.layers import Concatenate, GRU, TimeDistributed, Dense, Dropout, Bidirectional, Lambda, RepeatVector
import numpy as np

from models.relation_mask import RM
from models.relation_pure_baseline import PureBaseline


def output_shape(input_shape):
    return [input_shape[0], input_shape[-1]]


class Baseline(BaseModel):

    def __init__(self, max_len=125, class_num=9, use_development_set=True):

        super(Baseline, self).__init__(use_development_set)

        self.max_len = max_len
        self.class_num = class_num
        self.name = "baseline"

    def build_model(self):
        sentence = Concatenate()([self.sen_embedding,
                                  self.sen_entity_type_embedding,
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
                              position_mt=self.test_position_mt, position_me=self.test_position_ma)
        log = open("../log/" + self.name + ".txt", 'a+', encoding='utf-8')
        for i in range(max_epoch):
            self.model.fit({'sentence_word': self.train_sentence_words_input,
                            'sentence_entity_type': self.train_sentence_entity_inputs,
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
                                          'sentence_entity_type': self.test_sentence_entity_inputs,
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


if __name__ == '__main__':

    s = Baseline(max_len=125, class_num=9, use_development_set=False)
    for i in range(5):
        s.compile_model()
        s.train_model(max_epoch=35)
