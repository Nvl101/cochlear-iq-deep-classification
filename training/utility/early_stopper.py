'''
early stop training if the model is not making progress

NOTE: try using mathematical modeling to predict the rise/fall
'''
from abc import abstractclassmethod

class EarlyStopper:
    @abstractclassmethod
    def step(self, *args, **kwargs):
        pass
    @abstractclassmethod
    def stop(self) -> bool:
        pass

class ValLoss(EarlyStopper):
    losses = []
    _prev_loss = None
    # check in previous {cooldown} losses in list, if there're more than 1 decrease
    cooldown: int
    delta: float
    tolerance: int
    target: float
    def __init__(
            self, cooldown: int = 3, delta: float = 0.01,
            tolerance: int = 2, target: int = 0.2):
        self.cooldown = cooldown
        self.delta = delta
        self.tolerance = tolerance
        self.target = target
    def step(self, validation_loss):
        self.losses.append(validation_loss)
    def stop(self) -> bool:
        '''
        return True means early stopping applies
        '''
        lst_cooldown = self.losses[-self.cooldown:]
        n_increases = 0
        # upon reaching target, stop training
        if self.losses[-1] <= self.target:
            return True
        # if last entry is not decreasing, then return False
        if len(self.losses) < 2:
            return False
        if self.losses[-1] + self.delta < self.losses[-2]:
            return False
        for i in range(len(lst_cooldown)):
            if lst_cooldown[i] + self.delta >= lst_cooldown[i - 1]:
                n_increases += 1
            if n_increases >= self.tolerance:
                return True
        return False
