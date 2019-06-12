import pickle
import keras.optimizers
from keras.engine import Model
from keras.layers import Embedding, Input
import numpy as np


class BaseModel:

    def __init__(self, use_development_set=True):

        # used dirs
        self.save_dir = "../saved_models/"
        self.dir = "../inputs/"
        self.embedding_dir = "../resource/embedding_matrix.pk"
        self.entity_embedding_dir = "../resource/entity_type_matrix.pk"
        self.index_ids_file = "label_idx_dict.pk"

        # some basic parameters of the model
        self.model = None
        self.max_len = 125
        # self.tri_max_len = 5
        self.num_words = 6820
        self.entity_type_num = 63
        self.name = "base_model"
        self.correction_factor = [0, 335, 97, 30, 4, 1, 0, 3, 0]

        # pre-trained embeddings and their parameters.
        self.embedding_matrix = BaseModel.load_pickle(self.embedding_dir)
        self.entity_embedding_matrix = BaseModel.load_pickle(self.entity_embedding_dir)
        self.embedding_trainable = False

        self.EMBEDDING_DIM = 200
        self.ENTITY_TYPE_VEC_DIM = 50
        self.TRIGGER_TYPE_VEC_DIM = 50
        self.POSITION_VEC_DIM = 50

        # inputs to the model
        self.use_development_set = use_development_set

        [self.train_sentence_words_input, self.train_sentence_entity_inputs, self.train_trigger_type,
         self.train_entity_type, self.train_position_t, self.train_position_a, self.train_position_mt,
         self.train_position_ma, self.train_labels] = self.load_data(train=True)

        [self.test_sentence_words_input, self.test_sentence_entity_inputs, self.test_trigger_type,
         self.test_entity_type, self.test_position_t, self.test_position_a, self.test_position_mt,
         self.test_position_ma, self.test_labels] = self.load_data(train=False)

        [self.dev_sentence_words_input, self.dev_sentence_entity_inputs, self.dev_trigger_type,
         self.dev_entity_type, self.dev_position_t, self.dev_position_a, self.dev_position_mt,
         self.dev_position_ma, self.dev_labels] = [None] * 9

        # if you want to use development set, this part can help you to split the development set from the train set.
        # not provide this yet.
        if self.use_development_set:
            self.split_train_set()

        # dict used to calculate the F1
        self.index_ids = BaseModel.load_pickle(self.dir + self.index_ids_file)

        [self.sen_input, self.sen_entity_type_input, self.position_t, self.position_a, self.position_mt,
         self.position_ma, self.trigger_type_input, self.entity_type_input] = [None] * 8

        [self.sen_embedding, self.sen_entity_type_embedding, self.position_t_embedding, self.position_a_embedding,
         self.position_mt_embedding, self.position_ma_embedding, self.trigger_type_embedding, self.entity_type_embedding] = [None] * 8

        self.output = None

    def build_model(self):
        pass

    def train_model(self):
        pass

    def predict(self):
        pass

    def save_model(self, file=""):
        self.model.save_weights(self.save_dir + file)

    def load_data(self, train=True):
        if train:
            path = self.dir + "train_"
        else:
            path = self.dir + "real_test_"

        rf = open(path + "sentence_words_input.pk", 'rb')
        sentence_words_input = pickle.load(rf)
        rf.close()

        rf = open(path + "sentence_entity_inputs.pk", 'rb')
        sentence_entity_inputs = pickle.load(rf)
        rf.close()

        rf = open(path + "entity_type.pk", 'rb')
        entity_type = pickle.load(rf)
        rf.close()

        rf = open(path + "trigger_type.pk", 'rb')
        trigger_type = pickle.load(rf)
        rf.close()

        rf = open(path + "position_t.pk", 'rb')
        position_t = pickle.load(rf)
        rf.close()

        rf = open(path + "position_a.pk", 'rb')
        position_a = pickle.load(rf)
        rf.close()

        rf = open(path + "position_mt.pk", 'rb')
        position_mt = pickle.load(rf)
        rf.close()

        rf = open(path + "position_ma.pk", 'rb')
        position_ma = pickle.load(rf)
        rf.close()

        rf = open(path + "labels.pk", 'rb')
        labels = pickle.load(rf)
        rf.close()

        return [sentence_words_input, sentence_entity_inputs, trigger_type, entity_type, position_t, position_a,
                position_mt, position_ma, labels]

    @staticmethod
    def load_pickle(file):
        rf = open(file, 'rb')
        embedding_matrix = pickle.load(rf)
        rf.close()
        return embedding_matrix

    def split_train_set(self):
        develop_doc_ids = self.load_pickle(self.dir + "development_doc_ids.pk")
        sen_doc_ids = self.load_pickle(self.dir + "train_sen_doc_ids.pk")

        train_index = []
        develop_index = []

        for index, doc_id in enumerate(sen_doc_ids):
            if doc_id in develop_doc_ids:
                develop_index.append(index)
            else:
                train_index.append(index)

        self.dev_word_inputs = self.train_word_inputs[develop_index]
        self.dev_entity_inputs = self.train_entity_inputs[develop_index]
        self.dev_labels = self.train_labels[develop_index]
        self.dev_attention_labels = self.train_attention_labels[develop_index]

        self.train_word_inputs = self.train_word_inputs[train_index]
        self.train_entity_inputs = self.train_entity_inputs[train_index]
        self.train_labels = self.train_labels[train_index]
        self.train_attention_labels = self.train_attention_labels[train_index]

        print(np.shape(self.train_word_inputs))
        print(np.shape(self.dev_word_inputs))

    def compile_model(self):
        inputs = [self.sen_input, self.sen_entity_type_input, self.position_t,
                  self.position_a, self.trigger_type_input, self.entity_type_input] = self.make_input()

        [self.sen_embedding, self.sen_entity_type_embedding, self.position_t_embedding, self.position_a_embedding,
         self.trigger_type_embedding, self.entity_type_embedding] = self.embedded()

        self.output = self.build_model()

        self.model = Model(inputs=inputs, outputs=self.output)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['acc'])
        print(self.model.summary())

    def make_input(self):

        inputs = [None] * 6

        inputs[0] = Input(shape=(self.max_len,), dtype='int32', name='sentence_word')
        inputs[1] = Input(shape=(self.max_len,), dtype='int32', name='sentence_entity_type')
        inputs[2] = Input(shape=(self.max_len,), dtype='int32', name='position_t')
        inputs[3] = Input(shape=(self.max_len,), dtype='int32', name='position_a')
        inputs[4] = Input(shape=(1,), dtype='int32', name='trigger_type')
        inputs[5] = Input(shape=(1,), dtype='int32', name='entity_type')

        return inputs

    def embedded(self):

        self.sentence_embedding_layer = Embedding(self.num_words + 2,
                                                  self.EMBEDDING_DIM,
                                                  weights=[self.embedding_matrix],
                                                  input_length=self.max_len,
                                                  trainable=False,
                                                  mask_zero=True)
        sentence_embedding = self.sentence_embedding_layer(self.sen_input)

        entity_embedding_layer = Embedding(self.entity_type_num + 2,
                                           self.ENTITY_TYPE_VEC_DIM,
                                           weights=[self.entity_embedding_matrix],
                                           input_length=self.max_len,
                                           trainable=True,
                                           mask_zero=True)
        entity_embedding = entity_embedding_layer(self.sen_entity_type_input)
        print(np.shape(entity_embedding))
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

        return [sentence_embedding, entity_embedding, position_t_embedding, position_a_embedding,
                trigger_type_embedding, entity_type_embedding]
