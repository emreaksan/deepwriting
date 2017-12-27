import time
import os
from tensorflow.python.client import timeline
import numpy as np

def get_model_dir_timestamp(base_path=None, prefix="", suffix="", connector="_"):
    """
    Creates a directory name based on timestamp.

    Args:
        base_path: path of parent directory.
        prefix:
        suffix:
        connector: one connector character between prefix, timestamp and suffix.

    Returns:

    """

    timestamp_dir_name = prefix+connector+str(int(time.time()))+connector+suffix
    if base_path:
        return os.path.abspath(os.path.join(base_path, timestamp_dir_name))
    else:
        return timestamp_dir_name


def create_tf_timeline(model_dir, run_metadata):
    """
    This is helpful for profiling slow Tensorflow code.

    Args:
        model_dir:
        run_metadata:

    Returns:

    """
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    timeline_file_path = os.path.join(model_dir,'timeline.json')
    with open(timeline_file_path, 'w') as f:
        f.write(ctf)

def sample_bivariate_normal(mu, sigma, rho, num_samples=1):
    """
    Given Bivariate normal distribution parameters (mu, sigma, rho) draws samples.

    Args:
        mu: 2D array.
        sigma: 2D array.
        rho: Scalar.

    Returns:
        2D samples.
    """
    s1, s2 = sigma
    cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    return np.random.multivariate_normal(mu, cov, num_samples)