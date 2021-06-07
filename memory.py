import os
import random
from collections import deque
import pickle
import numpy as np

import config as cfg
import logger as lg


def load_memories(path='./memories/'):
    lg.logger_memory.info("LOADING MEMORIES FROM STORAGE")
    try:
        with open(path + 'dataset.pkl', 'rb') as f:
            memories = pickle.load(f)
        lg.logger_memory.info("SIZE OF LOADED MEMORIES: {:0>5d}".format(len(memories)))
        return memories
    except FileNotFoundError:
        return None


def compact_memories(path='./memories/'):
    memories = pickle.load(open(path + 'dataset.pkl', 'rb'))
    for fname in os.listdir(path):
        if fname.endswith('pkl') and not fname.startswith('dataset'):
            with open(path + fname, 'rb') as f:
                memories.extend(pickle.load(f))
            os.remove(path + fname)
    pickle.dump(memories, open(path + 'dataset.pkl', 'wb'))


class Memory:

    def __init__(self, size_short=cfg.MAX_STEPS, size_long=cfg.MEMORY_SIZE, ltmemory=None):
        if ltmemory is not None:
            self.ltmemory = ltmemory
        else:
            self.ltmemory = deque(maxlen=size_long)
        self.stmemory = deque(maxlen=size_short+1)
        self.logger = lg.setup_logger('memory', 'logs/memory.log')

    def __len__(self):
        return len(self.ltmemory)

    def commit_stmemory(self, fragment, data_augmentation=None):
        """
        :param fragment: dictionary to save
        :param data_augmentation: function to be applied in order to do data augmentation
        """
        if data_augmentation is not None:
            pass

        self.stmemory.append(fragment)

    def commit_ltmemory(self):
        self.logger.info('COMMITTING THE WHOLE EPISODE')
        for mem in self.stmemory:
            self.ltmemory.append(mem)
        self.clear_stmemory()

    def clear_stmemory(self):
        self.logger.info('CLEANING SHORT TERM MEMORY')
        self.stmemory.clear()

    def sample(self, batch_size=cfg.BATCH_SIZE):
        return random.sample(list(self.ltmemory), batch_size)

    def save(self, name):
        pickle.dump(self.ltmemory, open('memories/mem{}.pkl'.format(name), 'wb'))

    def clear_ltmemory(self):
        self.logger.info('CLEANING LONG TERM MEMORY')
        self.ltmemory.clear()

    def last_episode(self):
        return self.stmemory
