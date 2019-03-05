import pickle

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
        self.index_ids_file = "tri_index_ids.pk"

        # some basic parameters of the model
        self.model = None
        self.max_len = 125
        self.tri_max_len = 5
        self.num_words = 6820
        self.entity_type_num = 63

        # pre-trained embeddings and their parameters.
        self.embedding_matrix = BaseModel.load_pickle(self.embedding_dir)
        self.entity_embedding_matrix = BaseModel.load_pickle(self.entity_embedding_dir)
        self.embedding_trainable = False
        self.EMBEDDING_DIM = 200
        self.ENTITY_TYPE_VEC_DIM = 50

        # inputs to the model
        self.use_development_set = use_development_set

        [self.train_sentence_words_input, self.train_sentence_entity_inputs, self.train_trigger_words,
         self.train_trigger_types, self.train_positions, self.train_labels] = self.load_data(train=True)

        [self.test_sentence_words_input, self.test_sentence_entity_inputs, self.test_trigger_words,
         self.test_trigger_types, self.test_positions, self.test_labels] = self.load_data(train=False)

        [self.dev_sentence_words_input, self.dev_sentence_entity_inputs, self.dev_trigger_words,
         self.dev_trigger_types, self.dev_positions, self.dev_labels] = [None] * 6

        # if you want to use development set, this part can help you to split the development set from the train set.
        if self.use_development_set:
            self.split_train_set()

        # dict used to calculate the F1
        # self.index_ids = BaseModel.load_pickle(self.dir + self.index_ids_file)

        [self.sen_input, self.entity_type_input, self.position_input,
         self.trigger_input, self.trigger_type_input] = [None] * 5

        [self.sen_embedding, self.entity_embedding, self.position_embedding,
         self.trigger_embedding, self.trigger_type_embedding] = [None] * 5

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
            path = self.dir + "test_"

        rf = open(path + "sentence_words_input.pk", 'rb')
        sentence_words_input = pickle.load(rf)
        rf.close()

        rf = open(path + "sentence_entity_inputs.pk", 'rb')
        sentence_entity_inputs = pickle.load(rf)
        rf.close()

        rf = open(path + "trigger_words.pk", 'rb')
        trigger_words = pickle.load(rf)
        rf.close()

        rf = open(path + "trigger_types.pk", 'rb')
        trigger_types = pickle.load(rf)
        rf.close()

        rf = open(path + "positions.pk", 'rb')
        positions = pickle.load(rf)
        rf.close()

        rf = open(path + "labels.pk", 'rb')
        labels = pickle.load(rf)
        rf.close()

        return sentence_words_input, sentence_entity_inputs, trigger_words, trigger_types, positions, labels

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
        self.sen_input, self.entity_type_input = self.make_input()
        self.sen_embedding, self.entity_embedding = self.embedded()

        self.output = self.build_model()

        inputs = [self.sen_input, self.entity_type_input]

        self.model = Model(inputs=inputs, outputs=self.output)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['acc'])

    def make_input(self):

        inputs = [None] * 5

        inputs[0] = Input(shape=(self.max_len,), dtype='int32', name='sentence_input')
        inputs[1] = Input(shape=(self.max_len,), dtype='int32', name='entity_type_input')
        inputs[2] = Input(shape=(self.max_len,), dtype='int32', name='position_input')
        inputs[3] = Input(shape=(self.tri_max_len,), dtype='int32', name='trigger_input')
        inputs[4] = Input(shape=(1,), dtype='int32', name='trigger_type_input')

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
        entity_embedding = entity_embedding_layer(self.entity_type_input)

        return [sentence_embedding, entity_embedding]
