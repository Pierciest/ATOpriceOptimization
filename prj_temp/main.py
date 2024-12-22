import numpy as np
from instances import *
from solvers import *
import json 

n_rooms = 10
hospital = Hospital(n_rooms)
print(hospital.occupation)

hospital.add_patient(3)
hospital.add_patient(3)

print(hospital.occupation)
hospital.remove_patient(3)
print(hospital.occupation)

with open("./settings/solver_setting.json") as f:
    solver_setting = json.load(
        f
    )
ga = Ga_Solver(solver_setting)
ga.solve()
