import numpy as np


def cube_vertices_normals_and_indices():
    vertices = [
        # Front Face
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
        # Back Face
        (0.5, -0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (0.5, 0.5, -0.5),
        # Top Face
        (-0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        # Bottom Face
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, -0.5, 0.5),
        (-0.5, -0.5, 0.5),
        # Right Face
        (0.5, -0.5, 0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (0.5, 0.5, 0.5),
        # Left Face
        (-0.5, -0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (-0.5, 0.5, -0.5),
    ]

    normals = [
        # Front Face
        (0, 0, 1),
        (0, 0, 1),
        (0, 0, 1),
        (0, 0, 1),
        # Back Face
        (0, 0, -1),
        (0, 0, -1),
        (0, 0, -1),
        (0, 0, -1),
        # Top Face
        (0, 1, 0),
        (0, 1, 0),
        (0, 1, 0),
        (0, 1, 0),
        # Bottom Face
        (0, -1, 0),
        (0, -1, 0),
        (0, -1, 0),
        (0, -1, 0),
        # Right Face
        (1, 0, 0),
        (1, 0, 0),
        (1, 0, 0),
        (1, 0, 0),
        # Left Face
        (-1, 0, 0),
        (-1, 0, 0),
        (-1, 0, 0),
        (-1, 0, 0),
    ]

    # Indices to form triangles for GL_TRIANGLES
    indices = [
        # Front Face
        0,
        1,
        2,
        2,
        3,
        0,
        # Back Face
        4,
        5,
        6,
        6,
        7,
        4,
        # Top Face
        8,
        9,
        10,
        10,
        11,
        8,
        # Bottom Face
        12,
        13,
        14,
        14,
        15,
        12,
        # Right Face
        16,
        17,
        18,
        18,
        19,
        16,
        # Left Face
        20,
        21,
        22,
        22,
        23,
        20,
    ]

    return vertices, normals, indices


def grid_vertices_normals_and_indices(n: int, m: int, cell_size: float):
    vertices = []
    for i in range(n):
        for j in range(m):
            vertices.append(
                (
                    i * cell_size,
                    j * cell_size,
                )
            )

    triangle_vertices = np.array(vertices, dtype=np.float32)
    triangle_vertices = triangle_vertices.reshape((n, m, 2))

    triangle_indices = []
    for i in range(n - 1):
        for j in range(m - 1):
            triangle_indices.extend(
                [
                    i * m + j,
                    i * m + j + 1,
                    (i + 1) * m + j + 1,
                    (i + 1) * m + j + 1,
                    (i + 1) * m + j,
                    i * m + j,
                ]
            )

    triangle_vertices = triangle_vertices.reshape((-1, 2))
    triangle_vertices -= np.array(
        [(n - 1) * cell_size / 2.0, (m - 1) * cell_size / 2.0], dtype=np.float32
    )
    triangle_indices = np.array(triangle_indices, dtype=np.uint32)
    triangle_normals = np.array([[0, 1, 0]] * len(triangle_vertices), dtype=np.float32)
    return triangle_vertices, triangle_normals, triangle_indices
