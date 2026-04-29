import numpy as np


def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q

    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def rotation_matrix_to_quaternion(R):
    """
    Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz].

    Parameters
    R : np.ndarray
        3x3 rotation matrix

    Returns
    np.ndarray
        quaternion [qw, qx, qy, qz]
    """

    r11, r12, r13 = R[0]
    r21, r22, r23 = R[1]
    r31, r32, r33 = R[2]

    trace = r11 + r22 + r33

    if trace > 0:
        s_val = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s_val
        qx = (r32 - r23) / s_val
        qy = (r13 - r31) / s_val
        qz = (r21 - r12) / s_val

    elif (r11 > r22) and (r11 > r33):
        s_val = np.sqrt(1.0 + r11 - r22 - r33) * 2
        qw = (r32 - r23) / s_val
        qx = 0.25 * s_val
        qy = (r12 + r21) / s_val
        qz = (r13 + r31) / s_val

    elif r22 > r33:
        s_val = np.sqrt(1.0 + r22 - r11 - r33) * 2
        qw = (r13 - r31) / s_val
        qx = (r12 + r21) / s_val
        qy = 0.25 * s_val
        qz = (r23 + r32) / s_val

    else:
        s_val = np.sqrt(1.0 + r33 - r11 - r22) * 2
        qw = (r21 - r12) / s_val
        qx = (r13 + r31) / s_val
        qy = (r23 + r32) / s_val
        qz = 0.25 * s_val

    q = np.array([qw, qx, qy, qz])

    return q / np.linalg.norm(q)


def rk4(f, x, dt, *args):
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)

    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def quat_multiply(q, p):
    """
    Mnożenie kwaternionów.
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = p
    """
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_rotate(q, v):
    """
    rotacja wektora v przez kwaternion q
    """
    v_quat = np.array([0, *v])

    return quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))[1:]


def euler_to_quaternion(roll, pitch, yaw):
    """
    w1, x1, y1, z1 = q
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)

    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)

    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quaternion_to_euler(q):
    """
    q = [w, x, y, z]
    zwraca: roll, pitch, yaw
    """
    w, x, y, z = q

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # gimbal lock
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def vee(M):
    return np.array(
        [
            M[2, 1],
            M[0, 2],
            M[1, 0],
        ]
    )
