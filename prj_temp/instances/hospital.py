import math

class Hospital():

    def __init__(self, n_rooms):
        self.n_rooms = n_rooms
        self.occupation = [0] * n_rooms
    
    def add_patient(self, idx_room):
        self.occupation[idx_room] += 1

    def remove_patient(self, idx_room):
        self.occupation[idx_room] -= 1

    def fitness(self):
        return math.abs(sum(self.occupation) - 2 * self.n_rooms) 
