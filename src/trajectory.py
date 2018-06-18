# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import segment
import numpy as np, pygmo as pg
from pathos.multiprocessing import ProcessingPool as Pool

class Trajectory(object):

    def __init__(self, dynamics):

        # dynamics
        self.dynamics = dynamics

        # parallel
        self.pool = Pool(8)

    def set_bounds(self, Tlb, Tub, slb, sub):

        # set state bounds
        self.slb, self.sub = np.array(slb, float), np.array(sub, float)

        # set duration bounds
        self.Tlb, self.Tub = float(Tlb), float(Tub)

        # compute number of segments
        self.nseg = len(self.slb) - 1

    @staticmethod # NOTE: static for multiprocessing!
    def parallel_propagate(segment):
        segment.propagate()
        return segment

    def propagate(self):
        self.segments = self.pool.map(lambda seg: seg.propagate(), self.segments)
        #[self.segments[i].propagate() for i in range(self.nseg)]

    def process_records(self):

        # combine segment records
        self.states = np.vstack(([seg.states for seg in self.segments]))
        self.controls = np.apply_along_axis(self.dynamics.pontryagin, 1, self.states)

        # adjust durations
        for i in range(self.nseg - 1):
            self.segments[i + 1].times += self.segments[i].T

        # update times
        self.times = np.hstack((seg.times for seg in self.segments))

    def get_nobj(self):
        return 1

    def get_nic(self):
        return self.nseg

    def gradient(self, z):
        return pg.estimate_gradient(self.fitness, z)

    def solve(self, otol=1e-5, zguess=None, seperate=False):

        if seperate:
            self.segments = self.pool.map(lambda seg: seg.solve(), self.segments)
            #[seg.solve() for seg in self.segments]

        else:
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
            if zguess is None:
                pop = pg.population(prob, 1)
            else:
                pop = pg.population(prob, 0)
                pop.push_back(zguess)
            pop = algo.evolve(pop)

            # extract soltution
            self.zopt = pop.champion_x
            self.fitness(self.zopt)

        # combine records
        self.process_records()

class Indirect(Trajectory):

    def __init__(self, dynamics):

        # initialise base
        Trajectory.__init__(self, dynamics)

    def set_bounds(self, Tlb, Tub, slb, sub):

        # set bounds
        Trajectory.set_bounds(self, Tlb, Tub, slb, sub)

        # instantiate indirect segments
        self.segments = [segment.Indirect(self.dynamics) for i in range(self.nseg)]

        # set segment bounds
        for i in range(self.nseg):
            self.segments[i].set_bounds(Tlb, Tub, slb[i], sub[i], slb[i+1], sub[i+1])

    def set(self, times, states, costates):

        for i in range(self.nseg):
            self.segments[i].set(
                times[i], states[i], times[i+1], states[i+1], costates[i]
            )

    def get_bounds(self):

        # lower bounds
        lb = np.hstack((
            np.full((self.nseg, 1), self.Tlb),
            self.slb[1:],
            np.full((self.nseg, self.dynamics.sdim), -100)
        )).flatten()
        lb = np.hstack((self.slb[0], lb))

        # upper bounds
        ub = np.hstack((
            np.full((self.nseg, 1), self.Tub),
            self.sub[1:],
            np.full((self.nseg, self.dynamics.sdim), 100)
        )).flatten()
        ub = np.hstack((self.sub[0], ub))

        # return bounds
        return (lb, ub)

    def decode(self, z):

        # extract durations, states, and costates
        T, s, l = np.hsplit(z[self.dynamics.sdim:].reshape((self.nseg, 1 + self.dynamics.sdim*2)), [1, 1 + self.dynamics.sdim])
        T = T.flatten()
        s = np.vstack((z[:self.dynamics.sdim], s))
        # node times
        t = np.hstack((0, T.cumsum()))
        return t, s, l


    def fitness(self, z):

        # extract durations, states, and costates
        t, s, l = self.decode(z)

        # set trajectory
        self.set(t, s, l)

        # propagate trajectories
        self.propagate()

        # constrait vector
        con = list()

        # equality constraints
        for i in range(self.nseg):

            # segment
            seg = self.segments[i]

            # final fullstate
            fsf = seg.states[-1]

            # final state
            sf = fsf[:self.dynamics.sdim]

            # compute mismatch
            con.append(sf - seg.sf)

        # inquality constraints
        con.append([t[i] - t[i+1] for i in range(self.nseg)])

        # return fitness
        return np.hstack(([1], *con))

        # enforce smoothness
        '''
        for i in range(self.nseg - 1):

            # 1st segment final costates
            lf = self.segments[i].states[-1, self.dynamics.sdim:]

            # 2nd segment initial costates
            l0 = self.segments[i].l0

            # compute mismatch
            ceq = np.hstack((ceq, l0 - lf))
        '''

    def get_nec(self):
        return self.dynamics.sdim*self.nseg# + self.dynamics.sdim*(self.nseg - 1)
