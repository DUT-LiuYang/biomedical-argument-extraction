import collections
import pickle

Interactions = collections.namedtuple('Interactions', ['e1s', 'e2s', 'type'])
data_set = collections.namedtuple('data_set', ['true_tri_labels', 'predicted_tri_labels', 'entity_labels', 'ids'])


class DataIntegration:

    def __init__(self, train):
        # ------------- dirs used in this part --------------- #
        self.resource_dir = "../resource/"
        self.data_dir = "../data/"
        if train:
            self.resource_dir += "train_"
            self.data_dir += "train_"
        else:
            self.resource_dir += "test_"
            self.data_dir += "test_"
        self.train = train
        # ---------------------------------------------------- #

    def initialization(self):
        """
        read data.
        :return: None
        """
        rf = open(self.data_dir + "trigger_inputs.pk", 'rb')
        true_tri_labels = pickle.load(rf)
        rf.close()

        predicted_tri_labels = None
        if self.train:
            rf = open(self.data_dir + "predicted_trigger_inputs.pk", 'rb')
            predicted_tri_labels = pickle.load(rf)
            rf.close()

        rf = open(self.data_dir + "entity_inputs.pk", 'rb')
        entity_labels = pickle.load(rf)
        rf.close()

        # ------------- get BIO based id label of the sentences -------------
        rf = open(self.data_dir + "trigger_offsets.pk", 'rb')
        trigger_offsets_ids = pickle.load(rf)
        rf.close()

        rf = open(self.data_dir + "entity_offsets.pk", 'rb')
        entity_offsets_ids = pickle.load(rf)
        rf.close()

        sen_num = len(entity_labels)
        for i in range(sen_num):
            tri_id_label, tri_offsets_ids = self.get_id_label_type(char_offsets=token_offsets[i],
                                                                   offsets=trigger_offsets_ids[0],
                                                                   ids=trigger_offsets_ids[1])

            entity_id_label, _ = self.get_id_label_type(char_offsets=token_offsets[i],
                                                        offsets=entity_offsets_ids[0],
                                                        ids=entity_offsets_ids[1])

        # -------------------------------------------------------------------



        return None

    def get_id_label_type(self, char_offsets, offsets, ids):
        """
        amazing code written by myself from 2 years ago.
        :param char_offsets:
        :param offsets:
        :param ids:
        :return:
        """
        label = ""
        j = 0
        signal = False
        offset_ids = {}

        if len(offsets) == 0:
            for i in range(len(char_offsets)):
                label += "O "
            return label, offset_ids

        for i in range(len(char_offsets)):
            if j < len(offsets):
                s1 = int(char_offsets[i].split("-")[0])
                e1 = int(char_offsets[i].split("-")[1])
                s2 = int(offsets[j].split("-")[0])
                e2 = int(offsets[j].split("-")[1])
                if s1 >= s2 and e1 <= e2:
                    if signal and e1 == e2:
                        label_type = "E-" + ids[j]
                        offset_ids[char_offsets[i]] = ids[j]
                        j += 1
                        signal = False
                    elif signal:
                        label_type = "I-" + ids[j]
                        offset_ids[char_offsets[i]] = ids[j]
                    elif e1 == e2 and s1 == s2:
                        label_type = "S-" + ids[j]
                        offset_ids[char_offsets[i]] = ids[j]
                        j += 1
                    else:
                        label_type = "B-" + ids[j]
                        offset_ids[char_offsets[i]] = ids[j]
                        signal = True
                    label += " " + label_type
                else:
                    label += " O"
            else:
                label += " O"
        # print(label)
        return label, offset_ids
