import collections
import pickle

Interactions = collections.namedtuple('Interactions', ['e1s', 'e2s', 'type'])
data_set = collections.namedtuple('data_set', ['true_tri_labels', 'predicted_tri_labels', 'entity_labels', 'ids', 'offsets'])


class DataIntegration:

    all_tri_type = ["Regulation", "Cell_proliferation", "Gene_expression", "Binding", "Positive_regulation",
                    "Transcription", "Dephosphorylation", "Development", "Blood_vessel_development",
                    "Catabolism", "Negative_regulation", "Remodeling", "Breakdown", "Localization",
                    "Synthesis", "Death", "Planned_process", "Growth", "Phosphorylation"]

    def __init__(self, train, label_idx=None):
        # ------------- dirs used in this part --------------- #
        self.resource_dir = "../resource/"
        self.data_dir = "../data/"
        if train:
            # self.resource_dir += "train_"
            self.data_dir += "train_"
        else:
            # self.resource_dir += "test_"
            self.data_dir += "test_"
        self.train = train
        # ---------------------------------------------------- #

        self.data = None
        self.interactions = None
        self.duplicated_dict = {}

        self.initialize_data()

        rf = open(self.data_dir + "interactions.pk", 'rb')
        self.interactions = pickle.load(rf)
        rf.close()

        rf = open(self.data_dir + "duplicated_ids.pk", 'rb')
        self.duplicated_dict = pickle.load(rf)
        rf.close()

        self.label_idx_dict, self.ids_label_dict = self.construct_interaction_dict(label_idx)
        self.trigger_argument_type_dict = self.construct_structure_dict()

    def initialize_data(self):
        """
        read data.
        :return: None
        """
        # BIO labels. no pad.
        rf = open(self.data_dir + "trigger_inputs.pk", 'rb')
        true_tri_labels = pickle.load(rf)
        rf.close()

        # BIO labels. pad for test set.
        predicted_tri_labels = None
        if not self.train:
            predicted_tri_labels = []
            rf = open(self.resource_dir + "F_81.59682899207247_24.txt", 'r', encoding='utf-8')
            while True:
                line = rf.readline()
                if line == "":
                    break
                predicted_tri_labels.append(line.strip("\n").strip(" ").split(" "))
            rf.close()

        # BIO labels. no pad.
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

        rf = open(self.data_dir + "offsets.pk", 'rb')
        offsets = pickle.load(rf)
        rf.close()

        sen_num = len(entity_labels)
        id_labels = []
        for i in range(sen_num):
            tri_id_label, tri_offsets_ids = self.get_id_label_type(char_offsets=offsets[i],
                                                                   offsets=trigger_offsets_ids[i][0],
                                                                   ids=trigger_offsets_ids[i][1])

            entity_id_label, _ = self.get_id_label_type(char_offsets=offsets[i],
                                                        offsets=entity_offsets_ids[i][0],
                                                        ids=entity_offsets_ids[i][1])

            id_label = []
            for tl, el in zip(tri_id_label, entity_id_label):
                if tl != "O":
                    id_label.append(tl)
                elif el != "O":
                    id_label.append(el)
                else:
                    id_label.append("O")
            id_labels.append(id_label)
        # -------------------------------------------------------------------

        self.data = data_set(true_tri_labels, predicted_tri_labels, entity_labels, id_labels, offsets)

        return None

    def get_id_label_type(self, char_offsets, offsets, ids):
        """
        amazing code written by myself from 2 years ago.
        :param char_offsets:
        :param offsets:
        :param ids:
        :return:
        """
        label = []
        j = 0
        signal = False
        offset_ids = {}

        if len(offsets) == 0:
            for i in range(len(char_offsets)):
                label.append("O")
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
                    label.append(label_type)
                else:
                    label.append("O")
            else:
                label.append("O")

        return label, offset_ids

    def construct_interaction_dict(self, label_idx):

        if label_idx is None:
            temp = 1
            label_idx_dict = {}
            label_sum_dict = {}
        else:
            temp = len(label_idx) + 1
            label_idx_dict = label_idx
            label_sum_dict = {}
            for key in label_idx_dict.keys():
                label_sum_dict[key] = 0

        ids_label_dict = {}

        # for document
        for e1s, e2s, types in zip(self.interactions[0], self.interactions[1], self.interactions[2]):
            # for sentence
            for e1, e2, label in zip(e1s, e2s, types):
                key = e1 + e2
                if label in label_idx_dict.keys():
                    ids_label_dict[key] = label_idx_dict[label]
                    label_sum_dict[label] += 1
                else:
                    label_idx_dict[label] = temp
                    ids_label_dict[key] = temp
                    label_sum_dict[label] = 1
                    temp += 1

        for key, value in label_idx_dict.items():
            print(key + " " + str(value))

        for key, value in label_sum_dict.items():
            print(key + " " + str(value))

        return label_idx_dict, ids_label_dict

    def construct_structure_dict(self):

        print("reading ")

        rf = open(self.resource_dir + "structure.txt", 'r', encoding='utf-8')

        entity_type_idx_dict = {}
        num = 0

        trigger_argument_type_dict = {}

        while True:
            line = rf.readline()
            if line == "":
                break
            if "ENTITY" in line[0:6]:
                temp = line.strip("\n").split(" ")[1]
                entity_type_idx_dict[temp] = num
                num += 1
            elif "EVENT" in line[0:6]:
                line = line.strip("\n").split("\t")
                temp = line[0].split()[1]

                if temp not in self.all_tri_type:
                    continue

                # ---
                type_set = set()
                line = line[1:]
                for record in line:
                    record = record.split("] ")[1].split(",")
                    for label in record:
                        type_set.add(label)

                trigger_argument_type_dict[temp] = type_set

        rf.close()

        return trigger_argument_type_dict


if __name__ == '__main__':
    di_train = DataIntegration(train=True)
    di_test = DataIntegration(train=False, label_idx=di_train.label_idx_dict)
