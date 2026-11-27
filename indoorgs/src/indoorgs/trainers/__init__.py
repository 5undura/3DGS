"""학습기 어댑터. 새 학습기는 base.TrainerAdapter 를 따르고 REGISTRY 에 추가."""
from .brush import Brush
from .lichtfeld import LichtFeld

REGISTRY = {"brush": Brush(), "lichtfeld": LichtFeld()}
