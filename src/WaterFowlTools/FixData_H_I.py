# This func is intended to fixing some errors/offset between the datasets
import glob
from re import L
raw_fix_pattern = {
    'nopi_d': ['nopi d', 'nbopi d'],
    'nopi_h': ['nopi h'],
    'nsho_d': ['nosh d', 'nsho d', 'nshod'],
    'nsho_h': ['nosh h', 'nsho h'],
    'mall_d': ['mall d'],
    'mall_h': ['mall h', 'mall j'],
    'gadw': ['gadw_', 'gadw', 'gadw d', 'gadw h'],
    'agwt': ['agwt', 'agwt.'],
    'unk': ['unk', 'unknown', 'u'],
    'bird': ['bird'],
    'amwi_h': ['amwi h', 'amiw h'],
    'amwi_d': ['amwi d', 'ammi d', 'amiw d']}
code_name_dict = {'nsho_d': 'Shoveler Male',
                  'nsho_h': 'Shoveler Female',
                  'nopi_d': 'Pintail Male',
                  'nopi_h': 'Pintail Female',
                  'mall_d': 'Mallard Male',
                  'mall_h': 'Mallard Female',
                  'amwi_d': 'American Widgeon Male',
                  'amwi_h': 'American Widgeon Female',
                  'gadw': 'Gadwall',
                  'agwt': 'Green-winged teal',
                  'unk': 'Unknown',
                  'bird': 'Bird_discarded'}
print(len(raw_fix_pattern), len(code_name_dict))
reversed_pattern = dict()
for key in raw_fix_pattern:
    for v in raw_fix_pattern[key]:
        reversed_pattern[v] = key
for txt_dir in glob.glob('/home/zt253/Downloads/new_bird_h/Bird_H/*_class.txt'):
    with open(txt_dir, 'r') as f:
        data = f.readlines()
    in_txt = []
    for line in data:
        line = line.split(',', 1)
        if (line[0] in reversed_pattern.keys()):
            rep = reversed_pattern[line[0]]
        else:
            rep = line[0]
        if (rep in code_name_dict.keys()):
            in_txt.append([code_name_dict[rep]+','+line[1]])
    with open(txt_dir, 'w') as f:
        for line in in_txt:
            f.writelines(line)
