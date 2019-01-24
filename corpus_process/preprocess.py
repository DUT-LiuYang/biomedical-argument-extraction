from keras.preprocessing.sequence import pad_sequences

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import pickle
import collections
import numpy as np

Interactions = collections.namedtuple('Interactions', ['e1s', 'e2s', 'type'])


class PreProcessor:
    """
    Now let's start the argument extraction.
    """
    all_tri_type = ["Regulation", "Cell_proliferation", "Gene_expression", "Binding", "Positive_regulation",
                    "Transcription", "Dephosphorylation", "Development", "Blood_vessel_development",
                    "Catabolism", "Negative_regulation", "Remodeling", "Breakdown", "Localization",
                    "Synthesis", "Death", "Planned_process", "Growth", "Phosphorylation"]
    max_len = 125

    def __init__(self, train=True):

        # ------------- dirs used in this part --------------- #
        self.resource_dir = "../resource/"
        self.output_dir = "../data/"
        if train:
            self.resource_dir += "train_"
            self.output_dir += "train_"
        else:
            self.resource_dir += "test_"
            self.output_dir += "test_"
        # ---------------------------------------------------- #

        # -------------- word and entity label --------------- #
        self.triggers = []
        self.entities = []
        # ---------------------------------------------------- #

        # -------------- entity and tri offsets -------------- #
        self.offsets = []
        self.trigger_offsets_ids = []
        self.entity_offsets_ids = []
        # ---------------------------------------------------- #

        # -------------- entity and tri offsets -------------- #
        self.interactions = []
        # ---------------------------------------------------- #

        self.inputs = []
        self.duplicated_tri = {}

    def read_labels(self):
        """
        read entity and triggers from the label files.
        :return: None
        """
        rf = open(self.resource_dir + "entity_type.txt", 'r', encoding='utf-8')
        while True:
            line = rf.readline()
            if line == "":
                break
            self.entities.append(line.strip("\n").strip(" ").split(" "))
        rf.close()

        rf = open(self.resource_dir + "label.txt", 'r', encoding='utf-8')
        while True:
            line = rf.readline()
            if line == "":
                break
            self.triggers.append(line.strip("\n").strip(" ").split(" "))
        rf.close()

        return None

    def read_offset_and_trigger_index(self):
        """
        Read the offset of each word and
        :return:
        """
        rf = open(self.resource_dir + "offset_id.txt", 'r', encoding='utf-8')
        while True:
            line = rf.readline()
            if line == "":
                break
            line = line.strip("\n").split("#")
            self.offsets.append(line[0].split(" "))

            for i in range(4):
                line[i] = line[i].split()

            self.trigger_offsets_ids.append([line[1], line[2]])
            # for offset, tri_id in zip(line[1], line[2]):
            #     self.trigger_offsets_ids.append([offset, tri_id])

            self.entity_offsets_ids.append([line[3], line[4]])
            # for offset, entity_id in zip(line[3], line[4]):
            #     self.entity_offsets_ids.append([offset, entity_id])

        rf.close()

    def read_interactions(self):
        rf = open(self.resource_dir + "interaction.txt", 'r', encoding='utf-8')
        while True:
            line = rf.readline()
            if line == "":
                break
            line = line.strip("\n").split("#")

            for i in range(3):
                line[i] = line[i].split()

            self.interactions.append(Interactions(line[0], line[1], line[2]))

        rf.close()

    def read_word_idx(self):
        rf = open(self.resource_dir + "input.txt", 'r', encoding='utf-8')

        while True:
            line = rf.readline()
            if line == "":
                break
            line = line.strip("\n").strip().split()

            line = [int(x) for x in line]

            self.inputs.append(line)

        rf.close()

    def read_duplicated_ids(self):
        rf = open(self.resource_dir + "duplicated.txt", 'r', encoding='utf-8')

        while True:
            line = rf.readline()
            if line == "":
                break

            line = line.strip("\n").strip().split("*")

            self.duplicated_tri[line[0]] = line[1].split("#")

        rf.close()

    def write_data(self):

        print("writing data to files...")

        wf = open(self.output_dir + "word_inputs.pk", 'wb')
        pickle.dump(np.array(self.inputs), wf)
        wf.close()

        wf = open(self.output_dir + "entity_inputs.pk", 'wb')
        pickle.dump(np.array(self.entities), wf)
        wf.close()

        wf = open(self.output_dir + "trigger_inputs.pk", 'wb')
        pickle.dump(np.array(self.triggers), wf)
        wf.close()

        wf = open(self.output_dir + "interactions.pk", 'wb')
        pickle.dump(np.array(self.interactions), wf)
        wf.close()

        wf = open(self.output_dir + "offsets.pk", 'wb')
        pickle.dump(np.array(self.offsets), wf)
        wf.close()

        wf = open(self.output_dir + "trigger_offsets.pk", 'wb')
        pickle.dump(np.array(self.trigger_offsets_ids), wf)
        wf.close()

        wf = open(self.output_dir + "entity_offsets.pk", 'wb')
        pickle.dump(np.array(self.entity_offsets_ids), wf)
        wf.close()

        wf = open(self.output_dir + "duplicated_ids.pk", 'wb')
        pickle.dump(np.array(self.duplicated_tri), wf)
        wf.close()

        print("finish.")

    def __call__(self, *args, **kwargs):
        # -------------- read data from files -------------- #
        self.read_labels()
        self.read_offset_and_trigger_index()
        self.read_interactions()
        self.read_word_idx()
        self.read_duplicated_ids()

        self.get_entity_inputs()
        self.get_trigger_inputs()

        self.write_data()

    @staticmethod
    def read_ids(file):
        rf = open(file, 'r', encoding='utf-8')
        type_idx = {}
        while True:
            line = rf.readline().strip("\n")
            if line == "":
                break
            line = line.split("\t")
            type_idx[line[0]] = int(line[1])
        rf.close()

        idx_type = ["_" for i in range(len(type_idx))]
        for word, index in type_idx.items():
            idx_type[index] = word

        return type_idx, idx_type

    # ========== use the read data to generate inputs of model ==========
    def get_entity_inputs(self):
        """
        convert the entity label to idx inputs.
        :return:
        """
        entity_ids_file = "../resource/entity_ids.txt"
        entity_class_idx, _ = self.read_ids(entity_ids_file)

        for index, line in enumerate(self.entities):
            self.entities[index] = [entity_class_idx[x] for x in line]

        k = int(entity_class_idx["O"])

        self.entities = pad_sequences(self.entities, maxlen=self.max_len, value=k, padding='post')

    def get_trigger_inputs(self):
        """
        convert the trigger label to idx inputs.
        :return:
        """
        trigger_ids_file = "../resource/tri_ids.txt"
        trigger_class_idx, _ = self.read_ids(trigger_ids_file)

        class_num = len(trigger_class_idx)

        for index, line in enumerate(self.triggers):
            self.triggers[index] = [trigger_class_idx[x] for x in line]

    # ===================================================================


if __name__ == '__main__':
    p_train = PreProcessor(train=True)
    p_test = PreProcessor(train=False)
    p_train()
    p_test()
