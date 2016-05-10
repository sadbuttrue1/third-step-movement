import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import ode
import scipy
from scipy.constants import G
from astropy.constants import M_earth, R_earth
import math

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
    J = parameters.J
    jx = J[0]
    jy = J[1]
    jz = J[2]
    r = np.array(q[0:2])
    v = np.array(q[2:4])
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
    k = G * (M_e + parameters.m)
    dx = - k * r / (np.linalg.norm(r) ** 3)
    result = v
    result = np.append(result, dx)
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
vx0 = 8000
q_0 = np.array([R + parameters.h, 0, 0, vx0, 0, 0, 0, 0, np.pi / 6, 0])
tk = parameters.T * 100

solver = ode(dq).set_integrator('dopri5', nsteps=1)
solver.set_initial_value(q_0, 0).set_f_params(parameters)

sol_t = []
sol_q = []
while solver.t < tk:
    solver.integrate(tk, step=True)
    sol_t.append(solver.t)
    sol_q.append(solver.y)

sol_t = np.array(sol_t).reshape((len(sol_t), 1))
sol_q = np.array(sol_q)

rx = sol_q[:, 0]
ry = sol_q[:, 1]

plt.plot(sol_t, rx, label="rx")
plt.plot(sol_t, ry, label="ry")
plt.grid()
plt.legend()

vx = sol_q[:, 2]
vy = sol_q[:, 3]

plt.figure()
plt.plot(sol_t, vx, label="vx")
plt.plot(sol_t, vy, label="vy")
plt.grid()
plt.legend()

wx = sol_q[:, 4]
wy = sol_q[:, 5]
wz = sol_q[:, 6]

plt.figure()
plt.plot(sol_t, wx, label='w_x')
plt.plot(sol_t, wy, label='w_y')
plt.plot(sol_t, wz, label='w_z')
plt.grid()
plt.legend()

psi = sol_q[:, 7]
theta = sol_q[:, 8]
phi = sol_q[:, 9]
plt.figure()
plt.plot(sol_t, psi, label="psi")
plt.plot(sol_t, theta, label="theta")
plt.plot(sol_t, phi, label="phi")
plt.grid()
plt.legend()

plt.show()

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
