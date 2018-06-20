# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np
from scipy.integrate import ode
from behaviour import Fallback, Sequence
from dynamics import Dynamics

class Mission(object):

    def __init__(self, origin, waypoints, dynamics):

        # origin
        self.O = np.array(origin, float)

        # waypoints
        self.W = np.array(waypoints, float)

        # number of waypoints
        self.N = len(self.W)

        self.V = float(0.1)

        # mission completion
        self.alpha = 0

        # waypoint index
        self.beta = 0

        # simulation done
        self.gamma = 0

        # waypoint tolerance
        self.eps = 1e-5

        # decision maker
        self.decide = Fallback([self.terminal_beta, self.update_beta])
        self.decide = Sequence([self.decide, self.update_alpha])
        self.decide = Sequence([Fallback([self.at_waypoint, self.to_waypoint]),self.decide])
        self.decide = Fallback([self.complete, self.decide])
        self.decide = Sequence([self.decide, Fallback([self.at_origin, self.to_origin]), self.done])

        # dynamics
        self.dynamics = dynamics

        # numerical integrator
        self.integrator = ode(
            lambda t, s: self.dynamics.eom_state(s, self.control(t, s)),
            lambda t, s: self.dynamics.eom_state_jac(s, self.control(t, s))
        )

        # configure integrator
        self.integrator.set_integrator('dop853', atol=1e-14, rtol=1e-14, nsteps=10000)

        # set synchroniser
        self.integrator.set_solout(self.sync)

        # set initial state and time
        self.integrator.set_initial_value(np.hstack((self.O, [0, 0])), 0)

    def simulate(self, t=1000):

        # reset records
        self.times = np.empty((1, 0), float)
        self.states = np.empty((0, self.dynamics.sdim), float)
        self.controls = np.empty((0, self.dynamics.udim), float)
        self.statuses = np.empty((0, len(self.decide.response.keys())), int)

        # integrate
        self.integrator.integrate(t)
        return self.states

    def sync(self, time, state):

        # append times and states
        self.times = np.append(self.times, time)
        self.states = np.vstack((self.states, state))
        self.controls = np.vstack((self.controls, self.control(time, state)))

        # sync behaviour tree
        self.decide()
        print(self.decide.response)
        self.statuses = np.vstack((self.statuses, list(self.decide.response.values())))
        if self.gamma is 1:
            return -1

    def control(self, time, state):

        if len(self.times) >= 2:

            # time step size
            dt = time - self.times[-1]

            # relative target position
            post = self.target - state[:2]
            # relative target distance
            dist = np.linalg.norm(post)
            # relative target direction
            dirt = post/dist

            # relative velocity
            #velt = self.V

            u = np.array([1, *dirt], float)

        else:
            u = np.array([0, 0, 0], float)

        #print(u)
        return u

    def terminal_beta(self):
        res = int(self.beta is self.N - 1)
        return res

    def update_beta(self):
        self.beta += 1
        return 2

    def update_alpha(self):
        self.alpha = 1
        return 2

    def at_waypoint(self):
        return int(np.linalg.norm(self.states[-1, :2] - self.W[self.beta]) <= self.eps)

    def to_waypoint(self):
        self.target = self.W[self.beta]
        return 2

    def complete(self):
        return int(self.alpha)

    def at_origin(self):
        return int(np.linalg.norm(self.states[-1, :2] - self.O) <= self.eps)

    def to_origin(self):
        self.target = self.O
        return 2

    def done(self):
        self.gamma = 1
        return 2



if __name__ == "__main__":

    # dynamics
    from dynamics import Dynamics
    sys = Dynamics(thrust=10, area=1)

    # controller
    #from controller import PID
    #con = PID(1, 1, 1)

    # origin & waypoints
    from farm import Farm
    env = Farm(5, 10, 10, 20, 5, 40, 50)
    wps = env.simple_coverage()
    org = np.array([env.dsx, env.dsy])

    # mission
    mis = Mission(org, wps, sys)

    mis.simulate(50000)
    print(mis.beta)

    # plotstate
    import matplotlib.pyplot as plt, seaborn as sb, pandas as pd

    fig, ax = plt.subplots(sys.sdim)
    for i in range(sys.sdim):
        ax[i].plot(mis.times[1:], mis.states[1:, i], "k.-")

    fig, ax = plt.subplots(1)
    ax.plot(mis.states[1:, 0], mis.states[1:, 1], "k-")
    ax.quiver(mis.states[1:,0], mis.states[1:,1], mis.controls[1:,1], mis.controls[1:,2], scale=None)
    ax.set_aspect('equal')
    env.plot(ax)

    fig, ax = plt.subplots(3)
    for i in range(3):
        ax[i].plot(mis.times[1:], mis.controls[1:, i], "k.-")

    fig, ax = plt.subplots(1)
    data = pd.DataFrame(mis.statuses, columns=list(mis.decide.response.keys()))
    sb.heatmap(data.T, cmap="YlGnBu", ax=ax)

    plt.show()
