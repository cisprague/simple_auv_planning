# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np

class Controller(object):

    def __init__(self):
        pass

    def set(self, time, position, velocity):

        # target state
        self.target = np.array(state, float)

        # relevant state indicies
        self.index = np.invert(np.isnan(self.target))

        # set time
        self.time = time

class PID(Controller):

    def __init__(self, kp, ki, kd):

        # initialise base
        Controller.__init__(self)

        # gains
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        # metrics
        self.error = float(0)
        self.integral = float(0)

    def sync(self, time, state):

        # error
        error = self.target[self.index] - state[self.index]
        error = np.linalg.norm(error)

        # integral
        dt = time - self.time
        self.integral = self.integral + error*dt
        derrivative = (error - self.error)/dt
        self.error = error

        # set time
        self.time = time


    def control(self, time, state):

        # error
        error = self.target[self.index] - state[self.index]
        ux, uy = error/np.linalg.norm(error)
        error = np.linalg.norm(error)
        #print(error)

        # integral
        dt = time - self.time
        integral = self.integral + error*dt
        derrivative = (error - self.error)/dt

        return np.array([self.kp*error + self.ki*integral + self.kd*derrivative, ux, uy], float)

if __name__ == "__main__":

    import numpy as np
    a = np.array([10, 3, None, None], float)
    print(a[np.isnan(a)])
