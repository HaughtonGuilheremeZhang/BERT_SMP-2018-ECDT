# -*- coding: utf-8 -*-
'''
    Guess a label in random as a base_line
'''
import sys
import importlib
importlib.reload(sys)
#sys.setdefaultencoding('utf8')
import random


labels = ['website', 'tvchannel', 'lottery', 'chat', 'match',
          'datetime', 'weather', 'bus', 'novel', 'video', 'riddle',
          'calc', 'telephone', 'health', 'contacts', 'epg', 'app', 'music',
          'cookbook', 'stock', 'map', 'message', 'poetry', 'cinemas', 'news',
          'flight', 'translation', 'train', 'schedule', 'radio', 'email']


def random_guess():
    index = random.randint(0, len(labels) - 1)
    return labels[index]

if __name__ == '__main__':
    import json
    dev_dct = json.load(open(sys.argv[1], encoding='utf-8'))
    rguess_dct = {}
    for it in dev_dct:
        rguess_dct[it] = {"query": dev_dct[it]['query'], "label": random_guess()}
    json.dump(rguess_dct, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False)