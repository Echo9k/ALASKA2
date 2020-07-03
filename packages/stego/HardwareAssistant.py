from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.distribute import get_strategy
from tensorflow.config import experimental_connect_to_cluster
from tensorflow.tpu.experimental import initialize_tpu_system
# from tensorflow_core.python.distribute.strategy_combinations import tpu_strategy


# Detect GPU and TPU
class DetectHardware:
    """help to know your GPU/TPU configuration"""

    def __init__(self):
        self.gpu = self.gpu_info()
        self.tpu = self.tpu_info()
        self.strategy = self.tpu_strategy()
        self.replicas = self.strategy.num_replicas_in_sync

    @staticmethod
    def gpu_info():
        gpu_info = !nvidia - smi
        gpu_info = '\n'.join(gpu_info)
        if gpu_info.find('failed') >= 0: return None
        else: return gpu_info

    @staticmethod
    def tpu_info():
        """Show the appropriate distribution strategy TPU detection.
         No parameters necessary if TPU_NAME environment variable is set — always the case on Kaggle."""
        try: return TPUClusterResolver()
        except ValueError: return None

    def tpu_strategy(self):
        if self.tpu:
            experimental_connect_to_cluster(self.tpu)
            initialize_tpu_system(self.tpu)
            return TPUStrategy(self.tpu)
        else:
            # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
            return get_strategy()

    def __str__(self):
        if self.gpu:
            print(f'GPU {self.gpu}')
        elif self.tpu:
            print(f'tpu: {self.tpu}, \nreplicas: {self.replicas}')
        else:
            print("Nor GPU, nor TPU")