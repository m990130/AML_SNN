try:
    from .slayer import spikeLayer as layer
except ModuleNotFoundError as e:
    print(e)
try:
    from .slayerLoihi import spikeLayer as loihi
except ModuleNotFoundError as e:
    print(e)
# from slayer import yamlParams as params
from .slayerParams import yamlParams as params
try:
    from .spikeLoss import spikeLoss as loss
except ModuleNotFoundError as e:
    print(e)
from .spikeClassifier import spikeClassifier as predict
from .quantizeParams import quantizeWeights as quantize
from .learningStats import learningStats, learningStat
from .slayerUpsampling import UpSampling2D
'''
This modules bundles various SLAYER PyTorch modules as a single package.
The complete module can be imported as
>>> import slayerSNN as snn
* The spikeLayer will be available as snn.layer
* The SLAYER Loihi layer will be available as snn.loihi
* The yaml-parameter reader will be availabe as snn.params
* The spike-loss module will be available as snn.loss
* The spike-classifier module will be available as snn.predict
* The spike-IO module will be available as snn.io
* The quantize module will be available as snn.quantize 
'''

__all__ = ["layer", "loihi", "params", "loss", "predict", "quantize", "learningStats", "learningStat", "UpSampling2D"]
