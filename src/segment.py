# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

from scipy.integrate import ode

class Segment(object):

    def __init__(self, dynamics):

        # assign dynamics
        self.dynamics = dynamics

        # numerical integrator
        self.integrator = ode(self.eom, self.eom_jac)

        # configure integrator
        self.integrator.set_integrator('dop853', atol=1e-10, rtol=1e-10)

        # set recorder
        self.integrator.set_solout(self.record)

    def record(self, time, state):

        # append times
        self.times = np.hstack((self.times, time))

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
        self.s0lb, self.s0ub = np.array(s0lb), np.array(s0ub)

        # set intial state bounds
        self.sflb, self.sfub = np.array(sflb), np.array(sfub)

    def set(self, t0, s0, tf, sf):

        # set times
        self.t0, self.tf = float(t0), float(tf)

        # set states
        self.s0 = np.array(s0, float), np.array(sf, float)

    def propagate(self):

        # reset records
        self.reset_records()

        # integrate
        self.integrator.integrate(self.tf)

        # return final state
        return self.states[-1]

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
        self.dim = self.dynamic.sdim*2

        # free time transversality
        self.freetime = bool(True)

        # intial state transversality
        self.tv0 = bool(False)

        # final state transversality
        self.tvf = bool(True)

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

    def set_bounds(self, Tlb, Tub, s0lb, s0ub, sflb, sfub):

        # set duration and state bounds
        Segment.set_bounds(self, Tlb, Tub, s0lb, s0ub, sflb, sfub)

        # free time transversality
        if Tlb == Tub:
            self.freetime = False
        else:
            self.freetime = True

        # initial state transversality conditions
        self.tv0l, self.tv0l = list(), list()
        for l, u in zip(s0lb, s0ub):
            if l == u:
                self.tv0l.append(False)
            else:
                self.tv0l.append(True)

        # initial state transversality conditions
        self.tvfl, self.tvfl = list(), list()
        for l, u in zip(sflb, sfub):
            if l == u:
                self.tvfl.append(False)
            else:
                self.tvfl.append(True)

    def set(self, t0, s0, tf, sf, l0):

        # set states and time
        Segment.set(self, t0, s0, tf, sf)

        # set initial costates
        self.l0 = np.array(l0, float)

        # set initial integrator state
        self.set_integrator(np.hstack((self.s0, self.l0)), self.t0)

    def set_transversality(self, tv0, tvf, tvH):

        # consider inital state transversality
        self.tv0 = bool(tv0)
        # consider final state transversality
        self.tvf = bool(tvf)
        # consider duration transversality
        self.tvH = bool(tvH)


    def mismatch(self):

        # extract final fullstate
        fsf = self.propagate()

        # extract final state
        sf = fsf[:self.dynamics.sdim]

        # compute state mismatch
        ceq = sf - self.sf

        # inital transversality constraints
        if self.tv0:
            ceq = np.hstack((ceq, self.l0[self.tv0l]))

        # final transversality conditions
        if self.tvf:
            # compute final costate
            lf = fsf[self.dynamics.sdim:]
            ceq = np.hstack((ceq, fsf[self.tvfl]))

        # free time transversality
        if self.tvH and self.freetime:
            # compute control
            uf = self.dynamics.control(fsf)
            # compute Hamiltonian
            ceq = np.hstack((ceq, self.dynamics.hamiltonian(fsf, uf)))

        return ceq

    def get_nobj(self):
        return int(1)

    def get_nec(self):
        nec = self.dynamics.sdim
        if tv0: nec += sum(self.tv0l)
        if tvf: nec += sum(self.tv0f)
        if tvH and self.freetime: nec += 1
        return int(nec)

    def get_bounds(self):
        lb = [self.Tlb, *self.s0lb, *self.sflb, *[-100]*self.dynamics.sdim]
        ub = [self.Tub, *self.s0ub, *self.sfub, *[100]*self.dynamics.sdim]
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
