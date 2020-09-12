import abc
import numpy as np

class DataContainer(abc.ABC):
    def __init__(self, num_pairs=1000):
        self.num_pairs = num_pairs
        self.is_initialized = 0
        self.counter = -1
        
    def create_data_set(self, pair_generator):
        for i in range(self.num_pairs):
            self.add_pair(i, pair_generator.generate_random_pair())
        self.is_initialized = 1
        
    @abc.abstractmethod
    def add_pair(self, index, pair):
        pass
    
    @abc.abstractmethod
    def get_pair(index):
        pass
    
    def __len__(self):
        if not self.is_initialized:
            raise BaseException('data set was not created')
        return self.num_pairs
    
    def __getitem__(self, key):
        if not self.is_initialized:
            raise BaseException('data set was not created')
        if key >= self.num_pairs:
            raise IndexError('index {} is out of range'.format(key))
        return self.get_pair(key)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.counter += 1
        if self.counter >= self.num_pairs:
            self.counter = -1
            raise StopIteration
        return self[self.counter]
    
    
class FileDataContainer(DataContainer):
    def __init__(self, num_pairs=1000, inp_dir='inputs\\', outp_dir='outputs\\'):
        self.inputs_dir = inp_dir
        self.outputs_dir = outp_dir
        super().__init__(num_pairs)
        
    def add_pair(self, index, pair):
        inp, outp = pair
        inp.tofile(self.inputs_dir + "{}.txt".format(index))
        outp.tofile(self.outputs_dir + "{}.txt".format(index))
        
    def get_pair(self, index):
        if not self.is_initialized:
            raise BaseException('data set was not created')
        inp = np.fromfile(self.inputs_dir + "{}.txt".format(index))
        outp = np.fromfile(self.outputs_dir + "{}.txt".format(index))
        return (inp, outp)
    
    