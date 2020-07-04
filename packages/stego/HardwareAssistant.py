from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.distribute import OneDeviceStrategy, MirroredStrategy
from tensorflow.config import experimental_connect_to_cluster
from tensorflow.tpu.experimental import initialize_tpu_system
from tensorflow.config import list_physical_devices
from tensorflow.python.client import device_lib


# from tensorflow_core.python.distribute.strategy_combinations import tpu_strategy


# Hardware information
def info(params=None):
    return device_lib.list_local_devices(params)


def gpu_info():
    return list_physical_devices('GPU')


def tpu_info(params=None):
    """ Detect hardware, return appropriate distribution strategy.

    No parameters necessary if TPU_NAME environment variable is set
    In  Kaggle the TPU_Name environment variable is set.
    """
    try:
        # self.tpu = TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        return TPUClusterResolver(params)
    except ValueError:
        return None


def default_strategy(in_cross_check=False):
    from tensorflow.distribute import get_strategy, in_cross_replica_context
    if in_cross_check:
        if in_cross_replica_context():
            return get_strategy()
    else:
        get_strategy()


# Detect GPU and TPU
class Rig:
    """help to know your GPU/TPU configuration"""

    def __init__(self):
        self.gpu = gpu_info()
        self.tpu = tpu_info()
        self.strategy = self.adaptive_strategy()
        self.replicas = self.strategy.num_replicas_in_sync

    # Strategy
    def gpu_strategy(self):
        if len(self.gpu) == 1:
            return OneDeviceStrategy(device="/gpu:0")
        else:
            return MirroredStrategy()

    def tpu_strategy(self):
        experimental_connect_to_cluster(self.tpu)
        initialize_tpu_system(self.tpu)
        return TPUStrategy(self.tpu)

    def adaptive_strategy(self):
        if self.tpu:
            return self.tpu_strategy()
        else:
            return self.gpu_strategy()

    # Display hardware environment
    def __str__(self):
        if self.gpu:
            print(f'GPU {self.gpu}')
        elif self.tpu:
            print(f'Running on TPU {self.tpu.master()}',
                  f'\nreplicas: {self.replicas}')
        else:
            print("No accelerator found")
