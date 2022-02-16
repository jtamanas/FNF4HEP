import numpy as np

class Galton():
    def __init__(self, n_steps, odds_right, n_dims=1, noise=False, sigma=0.1):
        self.n_steps = n_steps
        self.n_dims = n_dims
        if odds_right > 1 or odds_right < 0:
            print("Odds of landing on the right must be less than 1 and greater than 0")
        self.odds_right = odds_right
        self.noise = noise
        self.sigma = sigma

    
    def simulate(self, n_balls):
        """
        Simulate the Galton board
        """
        #initialize position of the balls at x=0
        pos = np.zeros((int(n_balls), self.n_dims))

        # np.random.seed(0)
        #multiply n_steps by 2 and divide pos by 2 to avoid empty bins (only odd or only even)
        for step in range(self.n_steps*2):
            #generate a random number between 0 and 1
            r = np.random.random((int(n_balls), self.n_dims), )
            #if the random number is less than the odds of landing on the right,
            #then the ball lands on the right
            pos[r < self.odds_right] += 1
            pos[r >= self.odds_right] -= 1

        if self.noise:
            if self.sigma is None:
                noise = np.random.normal(0, 2/np.sqrt(self.n_steps), (int(n_balls), self.n_dims))
            else:
                noise = np.random.normal(0, self.sigma, (int(n_balls), self.n_dims))
            pos += noise

        return pos.squeeze()/2.0/np.sqrt(self.n_steps)
