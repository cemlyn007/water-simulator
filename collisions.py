import typing
import numpy as np


class Sphere(typing.NamedTuple):
    center: np.ndarray[any, np.float32]
    radius: np.ndarray[any, np.float32]


class Rectanguloid(typing.NamedTuple):
    corner0: np.ndarray[any, np.float32]
    corner1: np.ndarray[any, np.float32]

    def get_faces(self) -> np.ndarray[any, np.float32]:
        faces = np.empty((6, 4, 3), dtype=np.float32)
        faces[0] = np.array(
            [
                [self.corner0[0], self.corner0[1], self.corner0[2]],
                [self.corner0[0], self.corner1[1], self.corner0[2]],
                [self.corner0[0], self.corner1[1], self.corner1[2]],
                [self.corner0[0], self.corner0[1], self.corner1[2]],
            ],
            dtype=np.float32,
        )
        faces[1] = np.array(
            [
                [self.corner1[0], self.corner0[1], self.corner0[2]],
                [self.corner1[0], self.corner1[1], self.corner0[2]],
                [self.corner1[0], self.corner1[1], self.corner1[2]],
                [self.corner1[0], self.corner0[1], self.corner1[2]],
            ],
            dtype=np.float32,
        )
        faces[2] = np.array(
            [
                [self.corner0[0], self.corner0[1], self.corner0[2]],
                [self.corner1[0], self.corner0[1], self.corner0[2]],
                [self.corner1[0], self.corner0[1], self.corner1[2]],
                [self.corner0[0], self.corner0[1], self.corner1[2]],
            ],
            dtype=np.float32,
        )
        faces[3] = np.array(
            [
                [self.corner0[0], self.corner1[1], self.corner0[2]],
                [self.corner1[0], self.corner1[1], self.corner0[2]],
                [self.corner1[0], self.corner1[1], self.corner1[2]],
                [self.corner0[0], self.corner1[1], self.corner1[2]],
            ],
            dtype=np.float32,
        )
        faces[4] = np.array(
            [
                [self.corner0[0], self.corner0[1], self.corner0[2]],
                [self.corner1[0], self.corner0[1], self.corner0[2]],
                [self.corner1[0], self.corner1[1], self.corner0[2]],
                [self.corner0[0], self.corner1[1], self.corner0[2]],
            ],
            dtype=np.float32,
        )
        faces[5] = np.array(
            [
                [self.corner0[0], self.corner0[1], self.corner1[2]],
                [self.corner1[0], self.corner0[1], self.corner1[2]],
                [self.corner1[0], self.corner1[1], self.corner1[2]],
                [self.corner0[0], self.corner1[1], self.corner1[2]],
            ],
            dtype=np.float32,
        )
        return faces


def collide_sphere_rectanguloid(
    sphere: Sphere, rectanguloid: Rectanguloid
) -> np.ndarray[any, np.float32]:
    rectanguloid_faces = rectanguloid.get_faces()

    intersections = np.tile(
        np.where(
            (
                (rectanguloid.corner0 <= sphere.center)
                & (sphere.center <= rectanguloid.corner1)
            ).all(),
            sphere.center,
            np.array([np.nan, np.nan, np.nan], dtype=np.float32),
        ),
        (len(rectanguloid_faces), 1),
    )

    for face_index, face in enumerate(rectanguloid_faces):
        v = sphere.center - face[0]
        n = np.cross(face[1] - face[0], face[2] - face[0])
        v_dot_n = np.dot(v, n)
        signed_distance = v_dot_n / np.linalg.norm(n)
        if np.abs(signed_distance) <= sphere.radius:
            intersection_to_plane = (
                sphere.center - signed_distance * n
            ) / np.linalg.norm(n)
            face_min = np.min(face, axis=0)
            face_max = np.max(face, axis=0)
            possible_face_intersection = np.minimum(face_max, intersection_to_plane)
            possible_face_intersection = np.maximum(
                face_min, possible_face_intersection
            )
            distance = np.linalg.norm(sphere.center - possible_face_intersection)
            intersections[face_index] = np.where(
                distance <= sphere.radius,
                possible_face_intersection,
                intersections[face_index],
            )
    return intersections
