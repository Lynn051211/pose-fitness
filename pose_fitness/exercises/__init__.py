"""动作模块"""

from .squat import Squat
from .pushup import Pushup
from .pullup import Pullup
from .plank import Plank

EXERCISES = {
    "1": Squat,
    "2": Pushup,
    "3": Pullup,
    "4": Plank,
}
