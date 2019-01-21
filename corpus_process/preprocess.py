try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import pickle


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

        self.offsets = []
        self.trigger_offsets_ids = []
        self.entity_offsets_ids = []

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

            for offset, tri_id in zip(line[1], line[2]):
                self.trigger_offsets_ids.append([offset, tri_id])

            for offset, entity_id in zip(line[3], line[4]):
                self.entity_offsets_ids.append([offset, entity_id])

        rf.close()
