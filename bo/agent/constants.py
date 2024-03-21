RED = '\033[91m'
GREEN = '\033[92m'
END = '\033[0m'


from enum import Enum

class MainState(Enum):
    ACTIVE = 1
    INACTIVE = 0

class RolloutState(Enum):
    ACTIVE = 1
    INACTIVE = 0

class RegionState(Enum):
    ACTIVE = 1
    INACTIVE = 0
    # MAIN = MainState
    # ROLLOUT = RolloutState