from pkg_resources import get_distribution, DistributionNotFound

from tomocg.radonusfft import *
from tomocg.solver_tomo import *
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass