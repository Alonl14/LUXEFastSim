import pandas as pd
import json
import sys
from trainerFactory import create_trainer

with open("Config/cfg_inner.json", 'r') as inner_file:
    cfg_inner = json.loads(inner_file.read())
with open("Config/cfg_outer.json", 'r') as outer_file:
    cfg_outer = json.loads(outer_file.read())

numEpochs = sys.argv[0]

inner_trainer = create_trainer(cfg_inner)
outer_trainer = create_trainer(cfg_outer)

inner_trainer.run()
outer_trainer.run()