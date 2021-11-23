import numpy as np

class Gen_Data_loader():
    def __init__(self, batch_size, max_seq_len):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.token_stream = []

    def create_batches(self, data_file, player_file):
        self.token_stream = self.load_data(data_file)
        self.player_stream= self.load_data(player_file)
        
        # comute number of batches
        self.num_batch = int(len(self.token_stream) / self.batch_size)
        
        # drop last samples
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.player_stream= self.player_stream[:self.num_batch* self.batch_size]

        # Split data
        self.sequence_batch = np.split(
                np.array(self.token_stream), self.num_batch, 0)
        self.player_batch = np.split(
                np.array(self.player_stream), self.num_batch, 0)
        
        self.pointer = 0

    def load_data(self, filename):
        data_stream = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.max_seq_len:
                    data_stream.append(parse_line)
        return data_stream

    def next_batch(self):
        ret1 = self.sequence_batch[self.pointer]
        ret2 = self.player_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret1, ret2

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

