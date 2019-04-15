import collections
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

Interactions = collections.namedtuple('Interactions', ['e1s', 'e2s', 'type'])
data_set = collections.namedtuple('data_set', ['true_tri_labels', 'predicted_tri_labels', 'entity_labels', 'ids', 'offsets', 'sentences'])


class DataIntegration:

    all_tri_type = ["Regulation", "Cell_proliferation", "Gene_expression", "Binding", "Positive_regulation",
                    "Transcription", "Dephosphorylation", "Development", "Blood_vessel_development",
                    "Catabolism", "Negative_regulation", "Remodeling", "Breakdown", "Localization",
                    "Synthesis", "Death", "Planned_process", "Growth", "Phosphorylation"]

    all_entity_type = ["Amino_acid", "Anatomical_system", "Cancer", "Cell", "Cellular_component",
                       "DNA_domain_or_region", "Developing_anatomical_structure", "Drug_or_compound",
                       "Gene_or_gene_product", "Immaterial_anatomical_entity", "Multi-tissue_structure",
                       "Organ", "Organism", "Organism_subdivision", "Organism_substance", "Pathological_formation",
                       "Protein_domain_or_region", "Simple_chemical", "Tissue"]

    arg_idx = {"O": 0}
    idx_arg = {0: "O"}
    arg_num = 1

    def __init__(self, train, label_idx=None):
        # ------------- dirs used in this part --------------- #

        self.resource_dir = "../resource/"

        self.output_dir = "../inputs/"
        self.data_dir = "../data/"
        if train:
            self.output_dir += "train_"
            self.data_dir += "train_"
        else:
            self.output_dir += "real_test_"
            self.data_dir += "test_"
        self.train = train
        # ---------------------------------------------------- #

        self.data = None
        self.interactions = None
        self.duplicated_dict = {}

        rf = open(self.data_dir + "interactions.pk", 'rb')
        self.interactions = pickle.load(rf)
        rf.close()

        rf = open(self.data_dir + "duplicated_ids.pk", 'rb')
        self.duplicated_dict = pickle.load(rf)
        rf.close()

        self.initialize_data()

        self.label_idx_dict, self.ids_label_dict = self.construct_interaction_dict(label_idx)
        self.label_idx_dict["O"] = 0

        self.trigger_argument_type_dict = self.construct_structure_dict()

        self.trigger_entity_type_idx = {}
        for i, type in enumerate(self.all_tri_type):
            self.trigger_entity_type_idx[type] = i

        i = len(self.trigger_entity_type_idx)
        for type in self.all_entity_type:
            self.trigger_entity_type_idx[type] = i
            i += 1

        print("=== " + str(len(self.trigger_entity_type_idx)) + " ===")

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

        rf = open(self.data_dir + "word_inputs.pk", 'rb')
        sentences = pickle.load(rf)
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
            # print(id_label)
        # -------------------------------------------------------------------

        # print(str(self.duplicated_dict.keys()))

        # -----
        for i, sen in enumerate(id_labels):
            signal = False

            j = 0
            # print(sen)
            while j < len(sen):
                # print(j)
                # print(sen[j])
            # for j, id in enumerate(sen):
                id = sen[j]
                if id == "O":
                    j += 1
                    continue

                if id[0] == "S":
                    offset = offsets[i][j]

                    if offset in self.duplicated_dict.keys():
                        signal = True
                        # print(offset)
                        index = id.find(".e")
                        sen_id = id[2:index]

                        duplicated_ids = self.duplicated_dict[offset]
                        for did in duplicated_ids:
                            # print(id_labels[i][j] + " " + sen_id)
                            if sen_id in did:
                                id_labels[i][j] += "*" + did
                    j += 1
                elif id[0] == "B":
                    # print(id)
                    k = j + 1
                    while k < len(sen) and (sen[k][0] == "I" or sen[k][0] == "E"):
                        k += 1
                    offset = offsets[i][j].split("-")[0] + "-" + offsets[i][k - 1].split("-")[1]
                    # print(offset)
                    if offset in self.duplicated_dict.keys():
                        signal = True

                        index = id.find(".e")
                        sen_id = id[2:index]

                        duplicated_ids = self.duplicated_dict[offset]
                        for did in duplicated_ids:
                            # print(did)
                            # print(id_labels[i][j] + " " + sen_id)
                            if sen_id in did:
                                t = j
                                while t < k:
                                    id_labels[i][t] += "*" + did
                                    t += 1
                    j = k
                else:
                    j += 1
                            # print(id_labels[i][j] + "+")
                    # print(id_labels[i][j])
            # if signal:
            #     print(id_labels[i])
        self.data = data_set(true_tri_labels, predicted_tri_labels, entity_labels, id_labels, offsets, sentences)

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

    def __call__(self, *args, **kwargs):

        if self.train:
            [position_t, position_a, trigger_type, entity_type, sentence_idx, labels] = self.construct_relation_dataset(index=0)
        else:
            [position_t, position_a, trigger_type, entity_type, sentence_idx, labels] = self.construct_relation_test_dataset(index=1)

        [sentence_words_input, sentence_entity_inputs, trigger_type,
         entity_type, position_t, position_a,
         position_mt, position_ma] = self.dataset2relation_inputs(position_t, position_a, trigger_type, entity_type, sentence_idx)

        wf = open(self.output_dir + "sentence_words_input.pk", 'wb')
        pickle.dump(sentence_words_input, wf)
        wf.close()

        wf = open(self.output_dir + "sentence_entity_inputs.pk", 'wb')
        pickle.dump(sentence_entity_inputs, wf)
        wf.close()

        wf = open(self.output_dir + "position_t.pk", 'wb')
        pickle.dump(position_t, wf)
        wf.close()

        wf = open(self.output_dir + "position_a.pk", 'wb')
        pickle.dump(position_a, wf)
        wf.close()

        wf = open(self.output_dir + "position_mt.pk", 'wb')
        pickle.dump(position_mt, wf)
        wf.close()

        wf = open(self.output_dir + "position_ma.pk", 'wb')
        pickle.dump(position_ma, wf)
        wf.close()

        wf = open(self.output_dir + "trigger_type.pk", 'wb')
        pickle.dump(trigger_type, wf)
        wf.close()

        wf = open(self.output_dir + "entity_type.pk", 'wb')
        pickle.dump(entity_type, wf)
        wf.close()

        wf = open(self.output_dir + "labels.pk", 'wb')
        pickle.dump(labels, wf)
        wf.close()

        wf = open("../inputs/label_idx_dict.pk", 'wb')
        pickle.dump(self.label_idx_dict, wf)
        wf.close()

    def construct_dataset(self, index=0):

        trigger_words = []
        trigger_types = []
        sentence_words = []
        positions = []
        labels = []

        sen_num = 0

        for tri_labels, entity_labels, ids, offsets in zip(self.data[index], self.data[2],
                                                           self.data[3], self.data[4]):
            for i, tri_label in enumerate(tri_labels):
                # print(tri_label)

                if tri_label == "O":
                    continue

                event = tri_label.split("-")[1]
                if event not in self.all_tri_type:
                    continue

                #  need generate a new line==============
                current_trigger_words = [i]
                current_trigger_types = event
                current_sentence_words = sen_num
                current_positions = [i]

                current_labels = ["O"] * 125

                tids = ids[i][2:].split("*")  # a list, contains all ids of current trigger candidate.

                for tid in tids:
                    for j, id in enumerate(ids):

                        if id == "O" or i == j:
                            # current_labels.append("O")  # ---
                            continue

                        temp = id[2:].split("*")

                        for aid in temp:
                            key = tid + aid
                            if key in self.ids_label_dict:
                                alabel = self.ids_label_dict[key]
                                current_labels[j] = id[:2] + str(alabel)
                                break

                # print(current_labels)

                if tri_label[0] == "I" or tri_label[0] == "E":
                    trigger_words[-1] = trigger_words[-1] + current_trigger_words
                    positions[-1] = positions[-1] + current_positions
                else:
                    trigger_words.append(current_trigger_words)
                    trigger_types.append(current_trigger_types)
                    sentence_words.append(current_sentence_words)
                    positions.append(current_positions)
                    labels.append(current_labels)

            sen_num += 1

        return trigger_words, trigger_types, sentence_words, positions, labels

    def dataset2inputs(self, trigger_words, trigger_types, sentence_words, positions, labels):

        rf = open(self.data_dir + "entity_index.pk", 'rb')
        entity_index = pickle.load(rf)
        rf.close()

        sentence_words_input = []
        sentence_entity_inputs = []

        max_trigger_len = -1

        trigger_type_idx = {}
        for i, type in enumerate(self.all_tri_type):
            trigger_type_idx[type] = i

        for i, type in enumerate(trigger_types):
            trigger_types[i] = trigger_type_idx[type]

        for current_trigger_words, sentence_index in zip(trigger_words, sentence_words):
            sentence_words_input.append(self.data[-1][sentence_index][:])
            sentence_entity_inputs.append(entity_index[sentence_index][:])

            if len(current_trigger_words) > max_trigger_len:
                max_trigger_len = len(current_trigger_words)

            for i, current_trigger_word in enumerate(current_trigger_words):
                current_trigger_words[i] = self.data[-1][sentence_index][i]
        print(labels[0])
        for line in labels:
            for i, label in enumerate(line):
                if label not in self.arg_idx.keys():
                    if self.train:
                        self.arg_idx[label] = self.arg_num
                        self.idx_arg[self.arg_num] = label
                        line[i] = [0] * 28
                        line[i][self.arg_num] = 1
                        self.arg_num += 1
                    else:
                        line[i] = [0] * 28
                        print("###" + label)
                else:
                    line[i] = [0] * 28
                    line[i][self.arg_idx[label]] = 1

        positions = self.get_position_inputs(positions)

        print(len(self.arg_idx))
        print("@@@" + str(self.arg_num))

        # ---- need check data type of each variable.
        print(np.shape(sentence_words_input))
        print(np.shape(sentence_entity_inputs))
        print(np.shape(trigger_types))
        print(np.shape(trigger_words))
        print(np.shape(positions))
        print(np.shape(labels))
        print("---------------------------------------")
        print(sentence_words_input[0])
        print(sentence_entity_inputs[0])
        print(trigger_types[0])
        print(trigger_words[0])
        print(labels[0][4])
        print(positions[0])

        print("max length of trigger word is " + str(max_trigger_len))
        trigger_words = pad_sequences(trigger_words, value=0, padding='post', maxlen=5)
        print(trigger_words[0])
        print(np.shape(trigger_words))

        return sentence_words_input, sentence_entity_inputs, trigger_words, trigger_types, positions, labels

    def get_position_inputs(self, positions):

        res = []

        for i, position in enumerate(positions):
            res.append([1] * 125)

            left = position[0]
            right = position[-1]

            for index in position:
                res[i][index] = 0

            for j in range(left):
                res[i][j] = left - j

            for j in range(124 - right):
                res[i][right + j + 1] = j + 1

        return res

    def construct_relation_dataset(self, index=0):
        position_t = []
        position_a = []

        trigger_type = []
        entity_type = []

        sentence_idx = []
        labels = []

        sen_num = 0

        arg_sum = [0] * 9

        print(self.data[3][22])
        print(self.data[index][22])
        print(self.data[2][22])
        for tri_labels, entity_labels, ids, offsets in zip(self.data[index], self.data[2],
                                                           self.data[3], self.data[4]):
            for i, tri_label in enumerate(tri_labels):

                if tri_label == "O" or tri_label[0] == "I" or tri_label[0] == "E":
                    continue

                event = tri_label.split("-")[1]
                if event not in self.all_tri_type:
                    continue

                tids = ids[i][2:].split("*")  # a list, contains all ids of current trigger candidate.

                for j, id in enumerate(ids):
                    if id == "O" or id[0] == "I" or id[0] == "E" or i == j:
                        continue

                    if tri_labels[j] != "O":
                        entity = tri_labels[j][2:]
                    else:
                        entity = entity_labels[j][2:]

                    if entity not in self.all_tri_type and entity not in self.all_entity_type:
                        continue

                    if entity not in self.trigger_argument_type_dict[event]:
                        continue

                    # labels.append([0] * 9)
                    label = 0

                    eids = id[2:].split("*")

                    for tid in tids:
                        for eid in eids:
                            key = tid + eid
                            if key in self.ids_label_dict:
                                label = self.ids_label_dict[key]
                                # current_labels[j] = id[:2] + str(alabel)
                                break

                    # label
                    temp = [0] * 9
                    temp[label] = 1
                    labels.append(temp[:])
                    arg_sum[label] += 1

                    # positions
                    temp = [i]
                    if tri_label[0] == "B":
                        k = i + 1
                        while ids[k][0] == "I" or ids[k][0] == "E":
                            temp.append(k)
                            k += 1

                    position_t.append(temp[:])

                    temp = [j]
                    if id[0] == "B":
                        k = j + 1
                        while ids[k][0] == "I" or ids[k][0] == "E":
                            temp.append(k)
                            k += 1

                    position_a.append(temp[:])

                    # types
                    trigger_type.append(tri_label[2:])
                    if tri_labels[j] != "O":
                        entity_type.append(tri_labels[j][2:])
                    else:
                        entity_type.append(entity_labels[j][2:])

                    sentence_idx.append(sen_num)

            sen_num += 1

        for num in arg_sum:
            print(num)

        return position_t, position_a, trigger_type, entity_type, sentence_idx, np.array(labels)

    def dataset2relation_inputs(self, position_t, position_a, trigger_type, entity_type, sentence_idx):
        rf = open(self.data_dir + "entity_index.pk", 'rb')
        entity_index = pickle.load(rf)
        rf.close()

        sentence_words_input = []
        sentence_entity_inputs = []

        max_trigger_len = max_entity_len = -1

        for i, type in enumerate(trigger_type):
            trigger_type[i] = [self.trigger_entity_type_idx[type]]
            entity_type[i] = [self.trigger_entity_type_idx[entity_type[i]]]
            sentence_words_input.append(self.data[-1][sentence_idx[i]][:])
            sentence_entity_inputs.append(entity_index[sentence_idx[i]][:])

            if len(position_t[i]) > max_trigger_len:
                max_trigger_len = len(position_t[i])

            if len(position_a[i]) > max_entity_len:
                max_entity_len = len(position_a[i])

        position_mt = self.get_position_mask(position_t)
        position_ma = self.get_position_mask(position_a)

        position_t = self.get_position_inputs(position_t)
        position_a = self.get_position_inputs(position_a)

        res = [np.array(sentence_words_input), np.array(sentence_entity_inputs), np.array(trigger_type),
               np.array(entity_type), np.array(position_t), np.array(position_a),
               np.array(position_mt), np.array(position_ma)]

        for x in res:
            print(np.shape(x))

        print(sentence_words_input[41])
        print(position_t[41])
        print(position_a[41])
        print(position_mt[41])
        print(position_ma[41])

        return res

    def get_position_mask(self, positions):
        res = []

        for position in positions:
            res.append([0] * 125)

            for p in position:
                res[-1][p] = 1

        return res

    def construct_relation_test_dataset(self, index=1):
        position_t = []
        position_a = []

        trigger_type = []
        entity_type = []

        sentence_idx = []
        labels = []

        sen_num = 0

        arg_sum = [0] * 9
        confusing_num = 0

        for tri_labels, entity_labels, ids, offsets in zip(self.data[index], self.data[2],
                                                           self.data[3], self.data[4]):
            for i, tri_label in enumerate(tri_labels):

                if tri_label == "O" or tri_label[0] == "I" or tri_label[0] == "E":
                    continue

                event = tri_label.split("-")[1]
                if event not in self.all_tri_type:
                    continue

                signal = False
                if ids[i] == "O":
                    signal = True
                else:
                    tids = ids[i][2:].split("*")  # a list, contains all ids of current trigger candidate.

                for j, e_label in enumerate(tri_labels):
                    if e_label == "O" or e_label[0] == "I" or e_label[0] == "E" or i == j:
                        continue

                    entity = tri_labels[j][2:]

                    if entity not in self.all_tri_type:
                        continue

                    if entity not in self.trigger_argument_type_dict[event]:
                        continue

                    label = 0

                    if ids[j] != "O":
                        eids = ids[j][2:].split("*")
                    else:
                        eids = None

                    if signal or eids is None:
                        confusing_num += 1
                        label = 0
                    else:
                        for tid in tids:
                            for eid in eids:
                                key = tid + eid
                                if key in self.ids_label_dict:
                                    label = self.ids_label_dict[key]
                                    # current_labels[j] = id[:2] + str(alabel)
                                    break

                    # label
                    temp = [0] * 9
                    temp[label] = 1
                    labels.append(temp[:])
                    arg_sum[label] += 1

                    # positions
                    temp = [i]
                    if tri_label[0] == "B":
                        k = i + 1
                        while ids[k][0] == "I" or ids[k][0] == "E":
                            temp.append(k)
                            k += 1

                    position_t.append(temp[:])

                    temp = [j]
                    if tri_labels[j][0] == "B":
                        k = j + 1
                        while tri_labels[k][0] == "I" or tri_labels[k][0] == "E":
                            temp.append(k)
                            k += 1

                    position_a.append(temp[:])

                    # types
                    trigger_type.append(tri_label[2:])
                    entity_type.append(tri_labels[j][2:])

                    sentence_idx.append(sen_num)

                for j, e_label in enumerate(entity_labels):
                    if e_label == "O" or e_label[0] == "I" or e_label[0] == "E" or i == j:
                        continue

                    entity = entity_labels[j][2:]

                    if entity not in self.all_entity_type:
                        continue

                    if entity not in self.trigger_argument_type_dict[event]:
                        continue

                    label = 0

                    eids = ids[j][2:].split("*")

                    if signal:
                        label = 0
                        confusing_num += 1
                    else:
                        for tid in tids:
                            for eid in eids:
                                key = tid + eid
                                if key in self.ids_label_dict:
                                    label = self.ids_label_dict[key]
                                    # current_labels[j] = id[:2] + str(alabel)
                                    break

                    # label
                    temp = [0] * 9
                    temp[label] = 1
                    labels.append(temp[:])
                    arg_sum[label] += 1

                    # positions
                    temp = [i]
                    if tri_label[0] == "B":
                        k = i + 1
                        while ids[k][0] == "I" or ids[k][0] == "E":
                            temp.append(k)
                            k += 1

                    position_t.append(temp[:])

                    temp = [j]
                    if tri_labels[j][0] == "B":
                        k = j + 1
                        while tri_labels[k][0] == "I" or tri_labels[k][0] == "E":
                            temp.append(k)
                            k += 1

                    position_a.append(temp[:])

                    # types
                    trigger_type.append(tri_label[2:])
                    entity_type.append(entity_labels[j][2:])

                    sentence_idx.append(sen_num)

            sen_num += 1

        for num in arg_sum:
            print(num)

        print("confusing number: " + str(confusing_num))

        return position_t, position_a, trigger_type, entity_type, sentence_idx, np.array(labels)


if __name__ == '__main__':
    di_train = DataIntegration(train=True)
    di_train()

    di_test = DataIntegration(train=False, label_idx=di_train.label_idx_dict)
    di_test()
