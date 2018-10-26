from .callbacks import TerminateOnMetricNaN
from .DataGenerator import DataGenerator
from .foreground_metrics import foreground_f1_score
from .ternausnet import ternausnet
from .weighted_losses import weighted_bce
from .utils import *

COLLECTS = ['Atlanta_nadir7_catid_1030010003D22F00',
            'Atlanta_nadir8_catid_10300100023BC100',
            'Atlanta_nadir10_catid_1030010003CAF100',
            'Atlanta_nadir13_catid_1030010002B7D800',
            'Atlanta_nadir14_catid_10300100039AB000',
            'Atlanta_nadir16_catid_1030010002649200',
            'Atlanta_nadir19_catid_1030010003C92000',
            'Atlanta_nadir21_catid_1030010003127500',
            'Atlanta_nadir23_catid_103001000352C200',
            'Atlanta_nadir25_catid_103001000307D800',
            'Atlanta_nadir27_catid_1030010003472200',
            'Atlanta_nadir29_catid_1030010003315300',
            'Atlanta_nadir30_catid_10300100036D5200',
            'Atlanta_nadir32_catid_103001000392F600',
            'Atlanta_nadir34_catid_1030010003697400',
            'Atlanta_nadir36_catid_1030010003895500',
            'Atlanta_nadir39_catid_1030010003832800',
            'Atlanta_nadir42_catid_10300100035D1B00',
            'Atlanta_nadir44_catid_1030010003CCD700',
            'Atlanta_nadir46_catid_1030010003713C00',
            'Atlanta_nadir47_catid_10300100033C5200',
            'Atlanta_nadir49_catid_1030010003492700',
            'Atlanta_nadir50_catid_10300100039E6200',
            'Atlanta_nadir52_catid_1030010003BDDC00',
            'Atlanta_nadir53_catid_1030010003CD4300',
            'Atlanta_nadir53_catid_1030010003193D00']

BGR_8BIT_THRESHOLD = 3000
