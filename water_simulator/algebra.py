import numpy as np


def update_orbit_camera_position(
    azimuth_radians: np.float32, elevation_radians: np.float32, radius: np.float32
) -> np.ndarray[any, np.float32]:
    x = radius * np.cos(azimuth_radians) * np.cos(elevation_radians)
    y = radius * np.sin(elevation_radians)
    z = radius * np.sin(azimuth_radians) * np.cos(elevation_radians)
    return np.array([x, y, z], dtype=np.float32)


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = np.array(center - eye)
    f = f / np.linalg.norm(f)

    u = np.array(up)
    u = u / np.linalg.norm(u)

    s = np.cross(f, u)
    s = s / np.linalg.norm(s)

    u = np.cross(s, f)

    result = np.identity(4)

    result[0, 0] = s[0]
    result[1, 0] = s[1]
    result[2, 0] = s[2]

    result[0, 1] = u[0]
    result[1, 1] = u[1]
    result[2, 1] = u[2]

    result[0, 2] = -f[0]
    result[1, 2] = -f[1]
    result[2, 2] = -f[2]

    result[3, 0] = -np.dot(s, eye)
    result[3, 1] = -np.dot(u, eye)
    result[3, 2] = np.dot(f, eye)

    return result


def perspective(
    fov: np.ndarray, aspect: np.ndarray, near: np.ndarray, far: np.ndarray
) -> np.ndarray:
    """
    Create a perspective projection matrix.

    :param fov: Field of view in the y direction, in degrees.
    :param aspect: Aspect ratio, defined as width divided by height.
    :param near: Distance from the viewer to the near clipping plane.
    :param far: Distance from the viewer to the far clipping plane.
    :return: A 4x4 perspective projection matrix.
    """
    f = 1.0 / np.tan(fov / 2)
    nf = 1 / (near - far)

    result = np.zeros((4, 4))

    result[0, 0] = f / aspect
    result[1, 1] = f
    result[2, 2] = (far + near) * nf
    result[2, 3] = -1.0
    result[3, 2] = (2 * far * near) * nf

    return result


def translate(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Apply a translation to a 4x4 matrix.

    :param matrix: The original 4x4 matrix.
    :param vector: A 3-element translation vector.
    :return: The translated 4x4 matrix.
    """
    translation_matrix = np.identity(4)
    translation_matrix[3, 0:3] = vector

    return np.dot(matrix, translation_matrix)


def scale(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Apply a scaling transformation to a 4x4 matrix.

    :param matrix: The original 4x4 matrix.
    :param vector: A 3-element scaling vector.
    :return: The scaled 4x4 matrix.
    """
    scaling_matrix = np.identity(4)
    scaling_matrix[0, 0] = vector[0]
    scaling_matrix[1, 1] = vector[1]
    scaling_matrix[2, 2] = vector[2]

    return np.dot(matrix, scaling_matrix)
