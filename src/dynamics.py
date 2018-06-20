# Christopher Iliffe Sprague
# christopher.iliffe.sprauge@gmail.com

import numpy as np

class Dynamics(object):

    def __init__(self, thrust=10, mass=25, area=0.5, CD=0.47, rho=997, theta=0, vf=0.1):

        # thrust [N]
        self.thrust = float(thrust)

        # mass [kg]
        self.mass = float(mass)

        # planaform area [m^2]
        self.A = float(area)

        # coefficient of drag
        self.CD = float(CD)

        # fluid density [kg/m^3]
        self.rho = float(rho)

        # fluid velocity direction [rad]
        self.theta = float(theta)

        # homotopy
        self.alpha = float(0)

        # bounded control
        self.bound = bool(True)

        # state dimensions
        self.sdim = int(4)

        # control dimensions
        self.udim = int(3)

    def eom_state(self, state, control):

        # extract state
        x, y, vx, vy = state

        # extract control
        ut, ux, uy = control

        # common subexpression elimination
        x0 = 1/self.mass
        x1 = self.thrust*ut*x0
        x2 = self.A*self.CD*self.rho*x0*np.sqrt(vx**2 + vy**2)/2

        # return state transition
        ds = np.array([vx, vy, ux*x1 - vx*x2, uy*x1 - vy*x2], float)
        return ds

    def eom_state_jac(self, state, control):

        # extract state
        x, y, vx, vy = state

        # extract control
        ut, ux, uy = control

        # common subexpression elimination
        x0 = 1/self.mass
        x1 = vx**2
        x2 = vy**2
        x3 = np.sqrt(x1 + x2)
        x4 = -self.A*self.CD*self.rho*x0*x3/2
        x5 = self.A*self.CD*self.rho*x0/(2*x3)
        x6 = -vx*vy*x5

        # return state transition Jacobian
        dds = np.array([
            [0, 0,           1,           0],
            [0, 0,           0,           1],
            [0, 0, -x1*x5 + x4,          x6],
            [0, 0,          x6, -x2*x5 + x4]
        ], float)
        return dds

    def eom_fullstate(self, fullstate, control):

        # extract fullstate
        x, y, vx, vy, lx, ly, lvx, lvy = fullstate

        # extract control
        ut, ux, uy = control

        # common subexpression elimination
        e0 = self.thrust*ut/self.mass

        # return fullstate transition
        return np.array([vx, vy, ux*e0, uy*e0, 0, 0, -lx, -ly], float)

    def eom_fullstate_jac(self, fullstate, control):

        # extract fullstate
        x, y, vx, vy, lx, ly, lvx, lvy = fullstate

        # extract control
        ut, ux, uy = control

        # return fullstate transition Jacobian
        return np.array([
            [0, 0, 1, 0,  0,  0, 0, 0],
            [0, 0, 0, 1,  0,  0, 0, 0],
            [0, 0, 0, 0,  0,  0, 0, 0],
            [0, 0, 0, 0,  0,  0, 0, 0],
            [0, 0, 0, 0,  0,  0, 0, 0],
            [0, 0, 0, 0,  0,  0, 0, 0],
            [0, 0, 0, 0, -1,  0, 0, 0],
            [0, 0, 0, 0,  0, -1, 0, 0]
        ], float)

    def pontryagin(self, fullstate):

        # extract fullstate
        x, y, vx, vy, lx, ly, lvx, lvy = fullstate

        # common subexpression elimination
        e0 = lvx**2
        e1 = lvy**2
        e2 = np.sqrt(e0 + e1)
        e3 = 1/e2

        # thrust power
        ut = e3*(-self.thrust*e0 - self.thrust*e1 + self.alpha*self.mass*e2)/(2*self.mass*(self.alpha - 1))

        # bounded control
        if self.bound:
            ut = min(max(ut, 0), 1)

        # thrust direction
        ux = -lvx*e3
        uy = -lvy*e3

        # return control vector
        return np.array([ut, ux, uy], float)

    def lagrangian(self, control):

        # extract control
        ut, ux, uy = control

        # return Lagrangian
        return self.alpha*ut + (1 - self.alpha)*ut**2

    def hamiltonian(self, fullstate, control):

        # extract fullstate
        x, y, vx, vy, lx, ly, lvx, lvy = fullstate

        # extract control
        ut, ux, uy = control

        # common subexpression elimination
        e0 = self.thrust*ut/self.mass

        # return hamiltonian
        return lvx*ux*e0 + lvy*uy*e0 + lx*vx + ly*vy + self.alpha*ut + ut**2*(-self.alpha + 1)
