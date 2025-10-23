import os 

#  Constants
MOM_KEYS = {0: 'mom_-1_0_0', 1: 'mom_-2_0_0', 2: 'mom_-3_0_0', 3: 'mom_0_-1_0', 4: 'mom_0_-2_0', 5: 'mom_0_-3_0', 6: 'mom_0_0_-1', 7: 'mom_0_0_-2', 8: 'mom_0_0_-3', 9: 'mom_0_0_0', 10: 'mom_0_0_1', 11: 'mom_0_0_2', 12: 'mom_0_0_3', 13: 'mom_0_1_0', 14: 'mom_0_2_0', 15: 'mom_0_3_0', 16: 'mom_1_0_0', 17: 'mom_2_0_0', 18: 'mom_3_0_0'}
MOM_KEYS_INV = {v: k for k, v in MOM_KEYS.items()}
BASE_PATH = os.path.abspath('/p/scratch/exotichadrons/exotraction')
FLAVOR_ORDER = {'light': 0, 'strange': 1, 'charm': 2}

MESON_NAME_MAP = {
    'light_light': 'pi',
    'light_charm': 'D',
    'light_strange': 'K',
    'charm_strange': 'Ds'
}

DI_MESON_NAME_MAP = {
    ('light_light', 'light_light'): 'pipi',
    ('light_charm', 'light_light'): 'Dpi',
    ('light_light', 'light_strange'): 'piK',
    ('light_charm', 'light_strange'): 'DK',
    ('charm_strange', 'light_light'): 'Dspi',
    ('charm_strange', 'light_strange'): 'DsK',
    ('light_strange', 'light_strange'): 'KK',
    ('light_strange', 'light_charm'): 'KD',
    ('light_light', 'light_charm'): 'piD',
}