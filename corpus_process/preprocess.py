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

    def read_labels(self):
        """
        read entity and triggers from the label files.
        :return: None
        """
        rf = open(self.resource_dir + "entity_type", 'r', encoding='utf-8')
        while True:
            line = rf.readline()
            if line == "":
                break
            self.entities.append(line.strip("\n").strip(" ").split(" "))
        rf.close()

        rf = open(self.resource_dir + "label", 'r', encoding='utf-8')
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
        rf = open(self.resource_dir + "offset_id", 'r', encoding='utf-8')
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
        rf = open(self.resource_dir + "interaction", 'r', encoding='utf-8')
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
        rf = open(self.resource_dir + "_input", 'r', encoding='utf-8')

        while True:
            line = rf.readline()
            if line == "":
                break
            line = line.strip("\n").strip().split()

            line = [int(x) for x in line]

            self.inputs.append(line)

        rf.close()

    def write_data(self):
        wf = open(self.output_dir + "word_idx.pk", 'wb')
        pickle.dump(np.array(self.inputs), wf)
        wf.close()

    def __call__(self, *args, **kwargs):
        self.read_labels()
        self.read_offset_and_trigger_index()
        self.read_interactions()
        self.read_word_idx()

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


if __name__ == '__main__':
    pass
