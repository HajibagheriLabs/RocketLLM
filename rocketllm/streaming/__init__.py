"""Moving weights from storage into the device.

Reading, staging and transferring live here. Nothing in this package decides *what* to move -- that
is the cache's job -- only how to move it without wasting the machine's time.
"""
from .transfer import HostStagingPool

__all__ = ["HostStagingPool"]
