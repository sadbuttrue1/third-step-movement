import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import ode
import scipy
from scipy.constants import G
from astropy.constants import M_earth, R_earth
import math
from astropy.units.si import sday

omega_earth = 2 * np.pi / sday.si.scale

M_e = M_earth.value
mu = G * M_e
R = R_earth.value


def value(val, error):
    result = val + error * Parameters.RANDOMS[Parameters.RANDOM_SAMPLE_NUMBER]
    Parameters.RANDOM_SAMPLE_NUMBER += 1
    return result


class Parameters:
    T_NAME = "T"
    m_NAME = "m"
    J_NAME = "J"
    L_NAME = "L"
    l_NAME = "l"
    D_NAME = "D"
    r_c_NAME = "r_c"
    p_NAME = "p"
    h_NAME = "h"
    VALUE_NAME = "value"
    ERROR_NAME = "error"
    RANDOM_MU = 0
    RANDOM_SIGMA = 1. / 3.
    RANDOM_SAMPLE_NUMBER = 0
    RANDOMS = np.random.normal(RANDOM_MU, RANDOM_SIGMA, 1000)

    def __init__(self, file):
        json_content = json.load(file)

        if Parameters.T_NAME in json_content:
            self.T = json_content[Parameters.T_NAME]
        else:
            raise ValueError("No T in json")

        if Parameters.L_NAME in json_content:
            self.L = json_content[Parameters.L_NAME]
        else:
            raise ValueError("No L in json")

        if Parameters.l_NAME in json_content:
            self.l = json_content[Parameters.l_NAME]
        else:
            raise ValueError("No l in json")

        if Parameters.D_NAME in json_content:
            self.D = json_content[Parameters.D_NAME]
        else:
            raise ValueError("No D in json")

        if Parameters.h_NAME in json_content:
            self.h = json_content[Parameters.h_NAME]
        else:
            raise ValueError("No h in json")

        if Parameters.m_NAME in json_content and Parameters.ERROR_NAME in json_content[
            Parameters.m_NAME] and Parameters.VALUE_NAME in json_content[Parameters.m_NAME]:
            m = json_content[Parameters.m_NAME]
            val = m[Parameters.VALUE_NAME]
            error = m[Parameters.ERROR_NAME]
            self.m = value(val, error)
        else:
            raise ValueError("No m in json or doesn't have value or error")

        if Parameters.J_NAME in json_content:
            J = json_content[Parameters.J_NAME]
            self.J = []
            rand = Parameters.RANDOMS[Parameters.RANDOM_SAMPLE_NUMBER]
            Parameters.RANDOM_SAMPLE_NUMBER += 1
            for j in J:
                if Parameters.VALUE_NAME in j and Parameters.ERROR_NAME in j:
                    val = j[Parameters.VALUE_NAME]
                    error = j[Parameters.ERROR_NAME]
                    self.J.append(val + rand * error)
                else:
                    raise ValueError("j doesn't have value or error")
            self.J = np.asarray(self.J)
        else:
            raise ValueError("No J in json")

        if Parameters.r_c_NAME in json_content:
            r_c = json_content[Parameters.r_c_NAME]
            self.r_c = []
            for r in r_c:
                if Parameters.VALUE_NAME in r and Parameters.ERROR_NAME in r:
                    val = r[Parameters.VALUE_NAME]
                    error = r[Parameters.ERROR_NAME]
                    self.r_c.append(value(val, error))
                else:
                    raise ValueError("r_c doesn't have value or error")
        else:
            raise ValueError("No r_c in json")

        if Parameters.p_NAME in json_content:
            p = json_content[Parameters.p_NAME]
            self.p = {}
            for t in p:
                if Parameters.VALUE_NAME in p[t] and Parameters.ERROR_NAME in p[t]:
                    val = p[t][Parameters.VALUE_NAME]
                    error = p[t][Parameters.ERROR_NAME]
                    self.p[float(t)] = value(val, error)
                else:
                    raise ValueError("p doesn't have value or error")

        else:
            raise ValueError("No p in json")


def dq(t, q, parameters: Parameters):
    r = np.array(q[0:2])
    v = np.array(q[2:4])
    k = G * (M_e + parameters.m)
    dx = - k * r / (np.linalg.norm(r) ** 3)
    result = v
    result = np.append(result, dx)

    J = parameters.J
    jx = J[0]
    jy = J[1]
    jz = J[2]
    w = np.array(q[4:7])
    psi = q[7]
    theta = q[8]
    phi = q[9]
    A = rotation_matrix_from_euler(psi, theta, phi)
    M = 3 * mu / (R ** 3) * np.asarray(
        [(jz - jy) * A[2, 1] * A[2, 2],
         (jx - jz) * A[2, 2] * A[2, 0],
         (jy - jx) * A[2, 0] * A[2, 1]]
    )
    w_d = M
    w_d[0] += (jy - jz) * w[1] * w[2]
    w_d[1] += (jz - jx) * w[2] * w[0]
    w_d[2] += (jx - jy) * w[0] * w[1]
    w_d = np.divide(w_d, J)

    b = np.asarray([w[0], w[1], w[2]])
    a = np.asarray(
        [
            [
                np.sin(theta) * np.sin(phi),
                np.cos(phi),
                0
            ],
            [
                np.sin(theta) * np.cos(phi),
                -np.sin(phi),
                0
            ],
            [
                np.cos(theta),
                0,
                1.
            ]
        ]
    )
    angles_d = np.linalg.solve(a, b)
    result = np.append(result, w_d)
    result = np.append(result, angles_d)
    return result


def rotation_matrix_from_euler(psi, theta, phi):
    return np.asarray(
        [
            [
                np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
                -np.cos(theta) * np.sin(psi) * np.cos(phi) - np.cos(phi) * np.sin(psi),
                np.sin(theta) * np.sin(psi)
            ],
            [
                np.sin(psi) * np.cos(phi) + np.cos(theta) * np.cos(psi) * np.sin(phi),
                np.cos(theta) * np.cos(phi) * np.cos(psi) - np.sin(phi) * np.sin(psi),
                -np.cos(phi) * np.sin(theta)
            ],
            [
                np.sin(theta) * np.sin(psi),
                np.cos(phi) * np.sin(theta),
                np.cos(theta)
            ]
        ]
    )


# def quaternion_from_rotation_matrix(a):
#     q = [np.sqrt((np.trace(a) + 1.) / 4.)]
#     for i in range(3):
#         if math.isclose(a[i, i] / 2., (np.trace(a) - 1.) / 4.):
#             q.append(0.)
#         else:
#             print(a[i, i] / 2. - (np.trace(a) - 1.) / 4.)
#             q.append(np.sqrt(a[i, i] / 2. - (np.trace(a) - 1.) / 4.))
#     return np.asarray(q)
#
#
# def rotation_matrix_from_quaternion(q):
#     return np.asarray(
#         [
#             [
#                 2 * (q[0] ** 2 + q[1] ** 2) - 1,
#                 2 * (q[1] * q[2] - q[0] * q[3]),
#                 2 * (q[1] * q[3] + q[0] * q[2])
#             ],
#             [
#                 2 * (q[1] * q[2] + q[0] * q[3]),
#                 2 * (q[0] ** 2 + q[2] ** 2) - 1,
#                 2 * (q[2] * q[3] - q[0] * q[1])
#             ],
#             [
#                 2 * (q[1] * q[3] - q[0] * q[2]),
#                 2 * (q[2] * q[3] + q[0] * q[1]),
#                 2 * (q[0] ** 2 + q[3] ** 2) - 1
#             ]
#         ]
#     )
#
#
# def euler_from_rotation_matrix(a):
#     thetac = a[2, 2]
#     theta = np.arccos(thetac)
#     thetas = np.sqrt(1 - thetac ** 2)
#     psi = np.arccos(- a[2, 1] / thetas)
#     phi = np.arccos(a[1, 2] / thetas)
#     return psi, theta, phi


parameters = Parameters(open("sample.json"))
vy0 = 8000
x0 = R + parameters.h
q_11_0 = np.array([x0, 0, 0, vy0, 0, 0, 0, 0, np.pi / 6, 0])
tk = parameters.T * 128

k = G * (parameters.m + M_e)
h = vy0 ** 2 - 2 * k / x0
c = x0 * vy0
f = np.sqrt(k ** 2 + h * c ** 2)
e = f / k
p = c ** 2 / k
nu = np.linspace(0, 2 * np.pi, num=20)
r = p / (1 + e * np.cos(nu))
x = r * np.cos(nu)
y = r * np.sin(nu)

solver_11 = ode(dq).set_integrator('dopri5', nsteps=1)
solver_11.set_initial_value(q_11_0, 0).set_f_params(parameters)

sol_t_11 = []
sol_q_11 = []
while solver_11.t < tk:
    solver_11.integrate(tk, step=True)
    sol_t_11.append(solver_11.t)
    sol_q_11.append(solver_11.y)

sol_t_11 = np.array(sol_t_11).reshape((len(sol_t_11), 1))
sol_q_11 = np.array(sol_q_11)

rx = sol_q_11[:, 0]
ry = sol_q_11[:, 1]

plt.plot(sol_t_11, rx, label="rx")
plt.plot(sol_t_11, ry, label="ry")
plt.grid()
plt.legend()

vx = sol_q_11[:, 2]
vy = sol_q_11[:, 3]

plt.figure()
plt.plot(sol_t_11, vx, label="vx")
plt.plot(sol_t_11, vy, label="vy")
plt.grid()
plt.legend()

plt.figure()
plt.plot(rx, ry, label="numeric")
plt.plot(x, y, label="analytic")
plt.grid()
plt.legend()

wx = sol_q_11[:, 4]
wy = sol_q_11[:, 5]
wz = sol_q_11[:, 6]

plt.figure()
plt.plot(sol_t_11, wx, label='w_x')
plt.plot(sol_t_11, wy, label='w_y')
plt.plot(sol_t_11, wz, label='w_z')
plt.grid()
plt.legend()

psi_11 = sol_q_11[:, 7]
theta_11 = sol_q_11[:, 8]
phi = sol_q_11[:, 9]
plt.figure()
plt.plot(sol_t_11, psi_11, label="psi")
plt.plot(sol_t_11, theta_11, label="theta")
plt.plot(sol_t_11, phi, label="phi")
plt.grid()
plt.legend()


def dr(t, q, parameters: Parameters, r0: float, omega_0: float):
    beta = r0 / omega_0
    J = parameters.J
    alpha = J[2] / J[0]
    p = q[0: 2]
    angle = q[2: 4]

    dp = np.zeros(2)
    dp[0] = (p[0] / np.tan(angle[1]) - alpha * beta / (np.sin(angle[1]))) * np.sin(angle[0]) - p[1] * np.cos(angle[0])
    dp[1] = - (p[0] ** 2) / (np.sin(angle[1]) ** 3) * np.cos(angle[1]) \
            + p[0] / (np.sin(angle[1]) ** 2) * np.cos(angle[0]) \
            + alpha * beta * p[0] * (1 + np.cos(angle[1]) ** 2) / (np.sin(angle[1]) ** 3) \
            - alpha * beta * (np.cos(angle[0])) / (np.sin(angle[1]) ** 2) * np.cos(angle[1]) \
            - alpha ** 2 * beta ** 2 / (np.tan(angle[1]) * (np.sin(angle[1]) ** 2)) \
            - 3 * (alpha - 1) * np.cos(angle[1]) * np.sin(angle[1])
    dangle = np.zeros(2)
    dangle[0] = p[0] / (np.sin(angle[1]) ** 2) - np.cos(angle[0]) / np.tan(angle[1]) \
                - alpha * beta * np.cos(angle[1]) / (np.sin(angle[1]) ** 2)
    dangle[1] = p[1] - np.sin(angle[0])

    result = dp
    result = np.append(result, dangle)
    return result


J = parameters.J
alpha = J[2] / J[0]
omega_0 = np.sqrt(mu / (R ** 3))
r0 = 0.002
print(r0)
print(mu / (R ** 3))
print(omega_0)
beta = r0 / omega_0
print(beta)
print(alpha)
print(alpha * beta)
print(omega_0 / alpha)
cos_psi_11_0 = - alpha * beta
p_theta_11_0 = np.sqrt(1 - cos_psi_11_0 ** 2)
theta_11_0 = np.pi / 2
p_psi_11_0 = 0
q_11_0 = np.array([p_psi_11_0, p_theta_11_0, np.arccos(cos_psi_11_0), theta_11_0])
cos_psi_12_0 = - alpha * beta
p_theta_12_0 = np.sqrt(1 - cos_psi_12_0 ** 2)
theta_12_0 = np.pi / 2 + 0.01
p_psi_12_0 = 0
q_12_0 = np.array([p_psi_12_0, p_theta_12_0, np.arccos(cos_psi_12_0), theta_12_0])
cos_psi_13_0 = - alpha * beta
p_theta_13_0 = np.sqrt(1 - cos_psi_13_0 ** 2)
theta_13_0 = np.pi / 2 + 0.02
p_psi_13_0 = 0
q_13_0 = np.array([p_psi_13_0, p_theta_13_0, np.arccos(cos_psi_13_0), theta_13_0])
theta_21_0 = np.pi / 2
sin_psi_21_0 = 0
p_theta_21_0 = 0
p_psi_21_0 = 0
q_21_0 = np.array([p_psi_21_0, p_theta_21_0, np.arcsin(sin_psi_21_0), theta_21_0])
theta_22_0 = np.pi / 2 + 0.01
sin_psi_22_0 = 0
p_theta_22_0 = 0
p_psi_22_0 = 0
q_22_0 = np.array([p_psi_22_0, p_theta_22_0, np.arcsin(sin_psi_22_0), theta_22_0])
theta_23_0 = np.pi / 2 + 0.02
sin_psi_23_0 = 0
p_theta_23_0 = 0
p_psi_23_0 = 0
q_23_0 = np.array([p_psi_23_0, p_theta_23_0, np.arcsin(sin_psi_23_0), theta_23_0])
psi_31_0 = 0
p_theta_31_0 = 0
sin_theta_31_0 = alpha * beta / (3. * alpha - 4.)
p_psi_31_0 = 3. * (alpha - 1.) * sin_theta_31_0 * np.sqrt(1. - sin_theta_31_0 ** 2)
print(sin_theta_31_0)
print(p_psi_31_0)
q_31_0 = np.array([p_psi_31_0, p_theta_31_0, psi_31_0, np.arcsin(sin_theta_31_0)])
psi_32_0 = 0 + 0.01
p_theta_32_0 = 0
sin_theta_32_0 = alpha * beta / (3. * alpha - 4.)
p_psi_32_0 = 3. * (alpha - 1.) * sin_theta_32_0 * np.sqrt(1. - sin_theta_32_0 ** 2)
q_32_0 = np.array([p_psi_32_0, p_theta_32_0, psi_32_0, np.arcsin(sin_theta_32_0)])
psi_33_0 = 0 + 0.1
p_theta_33_0 = 0
sin_theta_33_0 = alpha * beta / (3. * alpha - 4.)
p_psi_33_0 = 3. * (alpha - 1.) * sin_theta_33_0 * np.sqrt(1. - sin_theta_33_0 ** 2)
q_33_0 = np.array([p_psi_33_0, p_theta_33_0, psi_33_0, np.arcsin(sin_theta_33_0)])
tk = 30

solver_11 = ode(dr).set_integrator('dopri5', nsteps=1)
solver_11.set_initial_value(q_11_0, 0).set_f_params(parameters, r0, omega_0)
solver_12 = ode(dr).set_integrator('dopri5', nsteps=1)
solver_12.set_initial_value(q_12_0, 0).set_f_params(parameters, r0, omega_0)
solver_13 = ode(dr).set_integrator('dopri5', nsteps=1)
solver_13.set_initial_value(q_13_0, 0).set_f_params(parameters, r0, omega_0)
solver_21 = ode(dr).set_integrator('dopri5', nsteps=1)
solver_21.set_initial_value(q_21_0, 0).set_f_params(parameters, r0, omega_0)
solver_22 = ode(dr).set_integrator('dopri5', nsteps=1)
solver_22.set_initial_value(q_22_0, 0).set_f_params(parameters, r0, omega_0)
solver_23 = ode(dr).set_integrator('dopri5', nsteps=1)
solver_23.set_initial_value(q_23_0, 0).set_f_params(parameters, r0, omega_0)
solver_31 = ode(dr).set_integrator('dopri5', nsteps=1)
solver_31.set_initial_value(q_31_0, 0).set_f_params(parameters, r0, omega_0)
solver_32 = ode(dr).set_integrator('dopri5', nsteps=1)
solver_32.set_initial_value(q_32_0, 0).set_f_params(parameters, r0, omega_0)
solver_33 = ode(dr).set_integrator('dopri5', nsteps=1)
solver_33.set_initial_value(q_33_0, 0).set_f_params(parameters, r0, omega_0)

sol_t_11 = []
sol_q_11 = []
sol_t_12 = []
sol_q_12 = []
sol_t_13 = []
sol_q_13 = []
sol_t_21 = []
sol_q_21 = []
sol_t_22 = []
sol_q_22 = []
sol_t_23 = []
sol_q_23 = []
sol_t_31 = []
sol_q_31 = []
sol_t_32 = []
sol_q_32 = []
sol_t_33 = []
sol_q_33 = []
while solver_11.t < tk:
    solver_11.integrate(tk, step=True)
    sol_t_11.append(solver_11.t)
    sol_q_11.append(solver_11.y)
while solver_12.t < tk:
    solver_12.integrate(tk, step=True)
    sol_t_12.append(solver_12.t)
    sol_q_12.append(solver_12.y)
while solver_13.t < tk:
    solver_13.integrate(tk, step=True)
    sol_t_13.append(solver_13.t)
    sol_q_13.append(solver_13.y)
while solver_21.t < tk:
    solver_21.integrate(tk, step=True)
    sol_t_21.append(solver_21.t)
    sol_q_21.append(solver_21.y)
while solver_22.t < tk:
    solver_22.integrate(tk, step=True)
    sol_t_22.append(solver_22.t)
    sol_q_22.append(solver_22.y)
while solver_23.t < tk:
    solver_23.integrate(tk, step=True)
    sol_t_23.append(solver_23.t)
    sol_q_23.append(solver_23.y)
tk = 1.58
while solver_31.t < tk:
    solver_31.integrate(tk, step=True)
    sol_t_31.append(solver_31.t)
    sol_q_31.append(solver_31.y)
while solver_32.t < tk:
    solver_32.integrate(tk, step=True)
    sol_t_32.append(solver_32.t)
    sol_q_32.append(solver_32.y)
while solver_33.t < tk:
    solver_33.integrate(tk, step=True)
    sol_t_33.append(solver_33.t)
    sol_q_33.append(solver_33.y)

sol_t_11 = np.array(sol_t_11).reshape((len(sol_t_11), 1))
sol_q_11 = np.array(sol_q_11)

p_psi_11 = sol_q_11[:, 0]
p_theta_11 = sol_q_11[:, 1]
psi_11 = sol_q_11[:, 2]
theta_11 = sol_q_11[:, 3]

sol_t_12 = np.array(sol_t_12).reshape((len(sol_t_12), 1))
sol_q_12 = np.array(sol_q_12)

p_psi_12 = sol_q_12[:, 0]
p_theta_12 = sol_q_12[:, 1]
psi_12 = sol_q_12[:, 2]
theta_12 = sol_q_12[:, 3]

sol_t_13 = np.array(sol_t_13).reshape((len(sol_t_13), 1))
sol_q_13 = np.array(sol_q_13)

p_psi_13 = sol_q_13[:, 0]
p_theta_13 = sol_q_13[:, 1]
psi_13 = sol_q_13[:, 2]
theta_13 = sol_q_13[:, 3]

sol_t_21 = np.array(sol_t_21).reshape((len(sol_t_21), 1))
sol_q_21 = np.array(sol_q_21)

p_psi_21 = sol_q_21[:, 0]
p_theta_21 = sol_q_21[:, 1]
psi_21 = sol_q_21[:, 2]
theta_21 = sol_q_21[:, 3]

sol_t_22 = np.array(sol_t_22).reshape((len(sol_t_22), 1))
sol_q_22 = np.array(sol_q_22)

p_psi_22 = sol_q_22[:, 0]
p_theta_22 = sol_q_22[:, 1]
psi_22 = sol_q_22[:, 2]
theta_22 = sol_q_22[:, 3]

sol_t_23 = np.array(sol_t_23).reshape((len(sol_t_23), 1))
sol_q_23 = np.array(sol_q_23)

p_psi_23 = sol_q_23[:, 0]
p_theta_23 = sol_q_23[:, 1]
psi_23 = sol_q_23[:, 2]
theta_23 = sol_q_23[:, 3]

sol_t_31 = np.array(sol_t_31).reshape((len(sol_t_31), 1))
sol_q_31 = np.array(sol_q_31)

p_psi_31 = sol_q_31[:, 0]
p_theta_31 = sol_q_31[:, 1]
psi_31 = sol_q_31[:, 2]
theta_31 = sol_q_31[:, 3]

sol_t_32 = np.array(sol_t_32).reshape((len(sol_t_32), 1))
sol_q_32 = np.array(sol_q_32)

p_psi_32 = sol_q_32[:, 0]
p_theta_32 = sol_q_32[:, 1]
psi_32 = sol_q_32[:, 2]
theta_32 = sol_q_32[:, 3]

sol_t_33 = np.array(sol_t_33).reshape((len(sol_t_33), 1))
sol_q_33 = np.array(sol_q_33)

p_psi_33 = sol_q_33[:, 0]
p_theta_33 = sol_q_33[:, 1]
psi_33 = sol_q_33[:, 2]
theta_33 = sol_q_33[:, 3]

import os
directory = "images"
if not os.path.exists(directory):
    os.mkdir("images")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(theta_11, p_theta_11, "+b", label="delta theta = 0")
ax.plot(theta_12, p_theta_12, "g", label="delta theta = 0.01")
ax.plot(theta_13, p_theta_13, "r", label="delta theta = 0.02")
ax.grid()
ax.legend()
ax.set_xlabel("theta")
ax.set_ylabel("p_theta")
fig.savefig("images/stat_1_theta.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(psi_11, p_psi_11, "+b", label="delta theta = 0")
ax.plot(psi_12, p_psi_12, "g", label="delta theta = 0.01")
ax.plot(psi_13, p_psi_13, "r", label="delta theta = 0.02")
ax.grid()
ax.legend()
ax.set_xlabel("psi")
ax.set_ylabel("p_psi")
fig.savefig("images/stat_1_psi.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(theta_21, p_theta_21, "+b", label="delta theta = 0")
ax.plot(theta_22, p_theta_22, "g", label="delta theta = 0.01")
ax.plot(theta_23, p_theta_23, "r", label="delta theta = 0.02")
ax.grid()
ax.legend()
ax.set_xlabel("theta")
ax.set_ylabel("p_theta")
fig.savefig("images/stat_2_theta.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(psi_21, p_psi_21, "+b", label="delta theta = 0")
ax.plot(psi_22, p_psi_22, "g", label="delta theta = 0.01")
ax.plot(psi_23, p_psi_23, "r", label="delta theta = 0.02")
ax.grid()
ax.legend()
ax.set_xlabel("psi")
ax.set_ylabel("p_psi")
fig.savefig("images/stat_2_psi.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(theta_31, p_theta_31, "+b", label="delta theta = 0")
ax.plot(theta_32, p_theta_32, "g", label="delta theta = 0.01")
ax.plot(theta_33, p_theta_33, "r", label="delta theta = 0.02")
ax.grid()
ax.legend(loc='upper left')
ax.set_xlabel("theta")
ax.set_ylabel("p_theta")
fig.savefig("images/stat_3_theta.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(psi_31, p_psi_31, "+b", label="delta theta = 0")
ax.plot(psi_32, p_psi_32, "g", label="delta theta = 0.01")
ax.plot(psi_33, p_psi_33, "r", label="delta theta = 0.02")
ax.grid()
ax.legend(loc='upper left')
ax.set_xlabel("psi")
ax.set_ylabel("p_psi")
fig.savefig("images/stat_3_psi.png")

# plt.show()

# def p(parameters: Parameters, t: float) -> float:
#     p_ = parameters.p
#     times = parameters.p.keys()
#     maximum = max(times)
#     minimum = min(times)
#     if t == maximum:
#         left_t = max({time: val for time, val in p_.items() if time < maximum})
#         right_t = min({time: val for time, val in p_.items() if time == maximum})
#         left_v = p_[left_t]
#         right_v = p_[right_t]
#     elif t > maximum:
#         # left_t = maximum
#         # right_t = maximum
#         # left_v = p_[maximum]
#         # right_v = p_[maximum]
#         return p_[maximum]
#     elif t == minimum:
#         left_t = max({time: val for time, val in p_.items() if time == minimum})
#         right_t = min({time: val for time, val in p_.items() if time > minimum})
#         left_v = p_[left_t]
#         right_v = p_[right_t]
#     else:
#         left_t = max({time: val for time, val in p_.items() if time <= t})
#         right_t = min({time: val for time, val in p_.items() if time > t})
#         left_v = p_[left_t]
#         right_v = p_[right_t]
#     a = np.array([[left_t, 1], [right_t, 1]])
#     b = np.array([left_v, right_v])
#     x = scipy.linalg.solve(a, b)
#     return x[0] * t + x[1]
#
#
# def n_p():
#     alpha = np.deg2rad(20)
#     beta = np.deg2rad(41)
#     gamma = np.deg2rad(49)
#     return np.array([-np.cos(alpha), -np.cos(beta) * np.sin(alpha), -np.cos(gamma) * np.sin(alpha)])
#
#
# def rho(parameters: Parameters):
#     x_c = parameters.r_c[0]
#     y_c = parameters.r_c[1]
#     z_c = parameters.r_c[2]
#     return np.array([parameters.L - parameters.l - x_c, np.cos(np.pi / 4.0) * parameters.D / 2.0 - y_c,
#                      np.sin(np.pi / 4.0) * parameters.D / 2.0 - z_c])
#
#
# def vector_to_quat(v):
#     v = np.array(v)
#     return np.append(0, v)
#
#
# def to_matrix(q):
#     return np.array([[q[0] + q[3] * 1j, -q[2] + q[1] * 1j], [q[2] + q[1] * 1j, q[0] - q[3] * 1j]])
#
#
# def from_matrix(q):
#     return np.array([q[0, 0].real, q[0, 1].imag, q[1, 0].real, q[0, 0].imag])
#
#
# def conjugate_matr(q):
#     # q = to_matrix(q)
#     return q.conj().T
#
#
# def rotate_quat_matrix(r, q):
#     r = vector_to_quat(r)
#     q = np.array(q)
#     r = to_matrix(r)
#     # print(q)
#     q = to_matrix(q)
#     return from_matrix(np.dot(np.dot(q, r), conjugate_matr(q)))[1:4]
#
#
# def dq(t, q, parameters: Parameters):
#     r = np.array(q[0:3])
#     v = np.array(q[3:6])
#     w = np.array(q[6:9])
#     quat = np.array(q[9:13])
#     P = np.multiply(n_p(), p(parameters, t))
#     Rho = rho(parameters)
#     M = np.cross(Rho, P)
#     F = rotate_quat_matrix(P, quat)
#     J = parameters.J
#     jx = J[0]
#     jy = J[1]
#     jz = J[2]
#
#     a = F / parameters.m
#
#     w_d = np.divide(M, J)
#     w_d[0] += (jy - jz) * w[1] * w[2] / jx
#     w_d[1] += (jz - jx) * w[2] * w[0] / jy
#     w_d[2] += (jx - jy) * w[0] * w[1] / jz
#
#     quat_d = [-w[0] * quat[1] - w[1] * quat[2] - w[2] * quat[3], w[0] * quat[0] + w[2] * quat[2] - w[1] * quat[3],
#               w[1] * quat[0] - w[2] * quat[1] + w[0] * quat[3], w[2] * quat[0] + w[1] * quat[1] - w[0] * quat[2]]
#     quat_d = np.array(quat_d) / 2
#
#     result = v
#     result = np.append(result, a)
#     result = np.append(result, w_d)
#     result = np.append(result, quat_d)
#     return result
#
#
# parameters = Parameters(open("sample.json"))
# q_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
# tk = parameters.T
#
# solver = ode(dq).set_integrator('dopri5', nsteps=1)
# solver.set_initial_value(q_0, 0).set_f_params(parameters)
#
# sol_t = []
# sol_q = []
# while solver.t < tk:
#     solver.integrate(tk, step=True)
#     sol_t.append(solver.t)
#     sol_q.append(solver.y)
#
# sol_t = np.array(sol_t).reshape((len(sol_t), 1))
# sol_q = np.array(sol_q)
# x = sol_q[:, 0]
# y = sol_q[:, 1]
# z = sol_q[:, 2]
#
# plt.plot(sol_t, sol_q[:, 6], label='w_x')
# plt.plot(sol_t, sol_q[:, 7], label='w_y')
# plt.plot(sol_t, sol_q[:, 8], label='w_z')
# plt.grid()
# plt.legend()
#
# plt.figure()
# plt.plot(sol_t, sol_q[:, 0], label='x')
# plt.plot(sol_t, sol_q[:, 1], label='y')
# plt.plot(sol_t, sol_q[:, 2], label='z')
# plt.grid()
# plt.legend()
#
# plt.figure()
# plt.plot(sol_t, sol_q[:, 3], label='v_x')
# plt.plot(sol_t, sol_q[:, 4], label='v_y')
# plt.plot(sol_t, sol_q[:, 5], label='v_z')
# plt.grid()
# plt.legend()
#
# plt.figure()
# plt.plot(sol_t, sol_q[:, 9], label='lambda_0')
# plt.plot(sol_t, sol_q[:, 10], label='lambda_1')
# plt.plot(sol_t, sol_q[:, 11], label='lambda_2')
# plt.plot(sol_t, sol_q[:, 12], label='lambda_3')
# plt.grid()
# plt.legend()
#
# plt.show()

# from mpl_toolkits.mplot3d import axes3d
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.plot(x, y, z)
# plt.show()
