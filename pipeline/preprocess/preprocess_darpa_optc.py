import os
import config
from pipeline.preprocess.preprocess_darpa_tc import preprocess_edge


def preprocess_darpa_optc(edge_file: str):
    edge = os.path.join(config.project_root, edge_file)
    dest = os.path.join(config.project_root, 'data/preprocess')
    preprocess_edge(edge, dest)
