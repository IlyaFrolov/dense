import numpy as np


class  InputOutputGenerator:
    def __init__(self):
        self.num_train_types = 3
        self.num_train_states = 4
        self.max_route_duration = 1000
        self.max_stop_duration = 60
        self.max_path_duration = 100
        self.num_segments = 100
        self.stations_number = 20
        self.output_dim = 500
        self.matrix_initiated = False
        self.train_density = 10
        self.matrix_name = "matrix.txt"
        
    def generate_random_route(self):
        result = np.zeros((self.stations_number, 2))
        current_time = 0
        delay = int(np.random.rand() * self.max_stop_duration)
        result[0, 0] = current_time
        current_time += delay
        result[0, 1] = current_time
        for i in range(1, self.stations_number):
            current_time += int(np.random.rand() * self.max_path_duration + 50)
            if np.random.rand() > 0.5:
                delay = int(np.random.rand() * self.max_stop_duration)
                result[i, 0] = current_time
                current_time += delay
                result[i, 1] = current_time
            else:
                result[i, :] = 0
        return result / self.max_route_duration
    
    def generate_random_segment(self):
        train_type = np.zeros(1).astype('int32')
        segment_condition = np.zeros(1).astype('int32')
        route = self.generate_random_route()
        return np.concatenate((train_type, segment_condition, route.flatten()))
    
    def generate_random_input(self):
        flag = 0
        result = self.generate_random_segment()
        for i in range(1, self.num_segments):
            flag = int(not (i % self.train_density) or 0)
            result = np.concatenate((result, self.generate_random_segment() * flag))
        return result
    
    def set_matrix(self, matrix=None, path='matrix.txt'):
        input_dim = (self.stations_number*2 + 2) * self.num_segments
        try:
            if not matrix:
                self.W = np.random.rand(self.output_dim, input_dim)*0.1
                self.W.tofile(path)
            else:
                self.W = matrix
            self.matrix_initiated = True
        except BaseException as er:
            self.matrix_initiated = False
            self.W = None
            raise er
            
    def generate_random_pair(self):
        if self.matrix_initiated:
            inp = self.generate_random_input()
            outp = np.dot(self.W, inp)
            return (inp, outp)
        else:
            raise BaseException('matrix is not initialized')
        
    def get_dimensions(self):
        input_dim = (self.stations_number*2 + 2) * self.num_segments
        return (input_dim, self.output_dim)
    

        