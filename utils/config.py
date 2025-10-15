from neuralforecast.models import NHITS, MLP, KAN
from neuralforecast.auto import AutoMLP, AutoNHITS, AutoKAN
from metaforecast.synth import (SeasonalMBB,
                                Jittering,
                                Scaling,
                                MagnitudeWarping,
                                TimeWarping,
                                DBA,
                                TSMixup)

ACCELERATOR = 'cpu'

MODELS = {
    'NHITS': NHITS,
    'MLP': MLP,
    'KAN': KAN,
    # 'AutoMLP': AutoMLP,
    # 'AutoNHITS': AutoNHITS,
    # 'AutoKAN': AutoKAN,
}

MODEL_CONFIG = {
    'AutoMLP': {},
    'AutoKAN': {},
    'AutoNHITS': {},
    'NHITS': {
        # 'start_padding_enabled': False,
        'accelerator': ACCELERATOR,
        'scaler_type': 'standard',
    },
    'MLP': {
        # 'start_padding_enabled': False,
        'accelerator': ACCELERATOR,
        'scaler_type': 'standard',
    },
    'KAN': {
        'accelerator': ACCELERATOR,
        'scaler_type': 'standard',
    },
}

SYNTH_METHODS = {
    'SeasonalMBB': SeasonalMBB,
    'Jittering': Jittering,
    'Scaling': Scaling,
    'TimeWarping': TimeWarping,
    'MagnitudeWarping': MagnitudeWarping,
    'TSMixup': TSMixup,
    'DBA': DBA,
}

SYNTH_METHODS_ARGS = {
    'SeasonalMBB': ['seas_period'],
    'Jittering': [],
    'Scaling': [],
    'MagnitudeWarping': [],
    'TimeWarping': [],
    'DBA': ['max_n_uids'],
    'TSMixup': ['max_n_uids', 'max_len', 'min_len']
}
    