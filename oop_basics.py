import numpy as np

class TwoDice:
    
    def __init__(self, d1=None, d2=None):
        if d1 is not None:
            self.x = d1
        else:
            self.x = np.random.randint(1,7)
        if d2 is not None:
            self.y = d2
        else:
            self.y = np.random.randint(1,7)
        self._target_number = np.random.randint(1,12)

    def __str__(self):
        return 'The dice are {} and {}.'.format(self.x,self.y)

    def roll(self):
        self.x = np.random.randint(1,7)
        self.y = np.random.randint(1,7)

    def is_success(self, target):
        return self.x + self.y >= target
    
    @property
    def test(self):
        return self.dsum >= self._target_number
    
    @property
    def dsum(self):
        return self.x + self.y
    
    @dsum.setter
    def dsum(self, num):
        if num < 7:
            self.x = np.random.randint(1,num)
        else:
            self.x = np.random.randint(num-6,7)
        self.y = num - self.x
