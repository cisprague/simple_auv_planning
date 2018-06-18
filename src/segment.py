# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

from scipy.integrate import ode
import numpy as np, pygmo as pg

class Segment(object):

    def __init__(self, dynamics):

        # assign dynamics
        self.dynamics = dynamics

        # numerical integrator
        self.integrator = ode(self.eom, self.eom_jac)

        # configure integrator
        self.integrator.set_integrator('dop853', atol=1e-10, rtol=1e-10, verbosity=1)

        # set recorder
        self.integrator.set_solout(self.record)

    def record(self, time, state):

        # append times
        self.times = np.append(self.times, time)

        # append states
        self.states = np.vstack((self.states, state))

    def reset_records(self):

        # reset time record
        self.times = np.empty((1, 0), float)

        # reset state record
        self.states = np.empty((0, self.dim), float)

    def set_bounds(self, Tlb, Tub, s0lb, s0ub, sflb, sfub):

        # set duration bounds
        self.Tlb, self.Tub = float(Tlb), float(Tub)

        # set intial state bounds
        self.s0lb, self.s0ub = np.array(s0lb, float), np.array(s0ub, float)

        # set intial state bounds
        self.sflb, self.sfub = np.array(sflb, float), np.array(sfub, float)

    def set(self, t0, s0, tf, sf):

        # set times
        self.t0, self.tf = float(t0), float(tf)

        # set states
        self.s0, self.sf = np.array(s0, float), np.array(sf, float)

    def propagate(self):

        # reset records
        self.reset_records()

        # integrate
        self.integrator.integrate(self.tf)

        # return final state
        return self.states[-1]

    def solve(self, otol=1e-5):

        # set optimisation params
        self.otol = otol

        # instantiate optimisation problem
        prob = pg.problem(self)

        # instantiate algorithm
        algo = pg.ipopt()
        algo.set_numeric_option("tol", self.otol)
        algo = pg.algorithm(algo)
        algo.set_verbosity(1)

        # instantiate and evolve population
        pop = pg.population(prob, 1)
        pop = algo.evolve(pop)

        # extract soltution
        self.zopt = pop.champion_x
        self.fitness(self.zopt)

        # process records
        self.process_records()

    def gradient(self, z):
        return pg.estimate_gradient(self.fitness, z)

    def plot(self, ax=None):

        # create axis
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        # plot states

class Direct(Segment):

    def __init__(self, dynamics):

        # initialise base
        Segment.__init__(self, dynamics)

        # state dimension
        self.dim = self.dynamics.sdim

    def eom(self, time, state, control):

        # state feedback controller
        if callable(control):
            control = control(state)

        # return state transition
        return self.dynamics.eom_state(state, control)

    def eom_jac(self, time, state, control):

        # state feedback controller
        if callable(control):
            control = control(state)

        # return state transition Jacobian
        return self.dynamics.eom_state_jac(state, control)

    def set(self, t0, s0, tf, sf):

        # set states and times
        Segment.set(self, t0, s0, tf, sf)

        # set intial integrator state
        self.integrator.set_intial_value(self.s0, self.t0)

    def mismatch(self):

        # return mismatch
        return self.propagate() - self.sf

class Indirect(Segment):

    def __init__(self, dynamics):

        # initialise base
        Segment.__init__(self, dynamics)

        # state dimensions
        self.dim = self.dynamics.sdim*2

    def eom(self, time, fullstate):

        # compute control
        control = self.dynamics.pontryagin(fullstate)

        # return fullstate transition
        return self.dynamics.eom_fullstate(fullstate, control)

    def eom_jac(self, time, fullstate):

        # compute control
        control = self.dynamics.pontryagin(fullstate)

        # return fullstate transition
        return self.dynamics.eom_fullstate_jac(fullstate, control)

    def set(self, t0, s0, tf, sf, l0):

        # set states and time
        Segment.set(self, t0, s0, tf, sf)

        # set initial costates
        self.l0 = np.array(l0, float)

        # set initial integrator state
        self.integrator.set_initial_value(np.hstack((self.s0, self.l0)), self.t0)

    def mismatch(self):

        # extract final fullstate
        fsf = self.propagate()

        # extract final state
        sf = fsf[:self.dynamics.sdim]

        # compute state mismatch
        ceq = sf - self.sf

        return ceq

    def get_nobj(self):
        return 1

    def get_nec(self):
        return self.dynamics.sdim

    def get_bounds(self):
        lb = np.array([self.Tlb, *self.s0lb, *self.sflb, *[-100]*self.dynamics.sdim], float)
        ub = np.array([self.Tub, *self.s0ub, *self.sfub, *[100]*self.dynamics.sdim], float)
        return (lb, ub)

    def fitness(self, z):

        # extract times
        t0, tf = 0, z[0]

        # extract states and costates
        s0, sf, l0 = z[1:].reshape((3, self.dynamics.sdim))

        # set segment
        self.set(t0, s0, tf, sf, l0)

        # compute mismatch
        ceq = self.mismatch()

        # return fitness
        return np.hstack(([1], ceq))

    def process_records(self):

        # controls
        self.controls = np.apply_along_axis(self.dynamics.pontryagin, 1, self.states)
