
class Ga_Solver():
    def __init__(self, solver_setting):
        self.n_individual = solver_setting['n_individual']
        self.mutation_prob = solver_setting['mutation_prob']
        self.n_generation = solver_setting['n_generation']
        self.time_limit = solver_setting['time_limit']

    def solve(self):
        pass
