import numpy as np

# ------------------------------------------------------------------------------
# 1. Define the Kinect v2 skeleton hierarchy
#    Each index corresponds to the joint (0..24).
#    -1 indicates no parent (root).
#    This is a commonly used parent setup for Kinect v2.
# ------------------------------------------------------------------------------

PARENTS = [
    -1,  # 0:  SpineBase        (root)
     0,  # 1:  SpineMid
     1,  # 2:  Neck
     2,  # 3:  Head
    20,  # 4:  ShoulderLeft
     4,  # 5:  ElbowLeft
     5,  # 6:  WristLeft
     6,  # 7:  HandLeft
    20,  # 8:  ShoulderRight
     8,  # 9:  ElbowRight
     9,  # 10: WristRight
    10,  # 11: HandRight
     0,  # 12: HipLeft
    12,  # 13: KneeLeft
    13,  # 14: AnkleLeft
    14,  # 15: FootLeft
     0,  # 16: HipRight
    16,  # 17: KneeRight
    17,  # 18: AnkleRight
    18,  # 19: FootRight
     1,  # 20: SpineShoulder
     7,  # 21: HandTipLeft
     7,  # 22: ThumbLeft
    11,  # 23: HandTipRight
    11,  # 24: ThumbRight
]

# ------------------------------------------------------------------------------
# 2. Utilities: rotation matrix from vector-to-vector, matrix to Euler, etc.
# ------------------------------------------------------------------------------

def rotation_from_vectors(v_from, v_to):
    """
    Returns a 3x3 rotation matrix that rotates the normalized vector v_from
    onto the normalized vector v_to (both 3D).
    """
    # Normalize inputs
    v_from = v_from / np.linalg.norm(v_from)
    v_to   = v_to   / np.linalg.norm(v_to)

    # Handle the degeneracy cases (vectors almost identical or opposite)
    dot = np.dot(v_from, v_to)
    if dot > 0.9999:
        # No (or negligible) rotation needed
        return np.eye(3)
    elif dot < -0.9999:
        # Rotating 180 degrees about any perpendicular axis
        # Find an orthogonal vector:
        orth = np.array([1, 0, 0])
        if abs(np.dot(v_from, orth)) > 0.9:
            orth = np.array([0, 1, 0])
        # Create an axis perpendicular to v_from
        axis = np.cross(v_from, orth)
        axis /= np.linalg.norm(axis)
        return rotation_from_axis_angle(axis, np.pi)

    # General case: use axis-angle
    axis = np.cross(v_from, v_to)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    return rotation_from_axis_angle(axis, angle)


def rotation_from_axis_angle(axis, angle):
    """
    Rodrigues' rotation formula to get the 3x3 rotation matrix
    for rotation about `axis` by `angle`.
    """
    ux, uy, uz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array([
        [c + ux*ux*C,     ux*uy*C - uz*s, ux*uz*C + uy*s],
        [uy*ux*C + uz*s,  c + uy*uy*C,    uy*uz*C - ux*s],
        [uz*ux*C - uy*s,  uz*uy*C + ux*s, c + uz*uz*C   ]
    ], dtype=np.float64)
    return R


def rotation_matrix_to_euler_xyz(R):
    """
    Convert a 3x3 rotation matrix to Euler angles about X, then Y, then Z.
    Returns angles in radians as (rx, ry, rz).
    
    The formula used (assuming R = Rx * Ry * Rz):
      ry = asin(-r13)
      rx = atan2(r23, r33)
      rz = atan2(r12, r11)
    where rIJ is the element at row I, column J of R (1-based indexing).
    """
    # Using row/column 0-based indexing in code:
    # R = [[r00, r01, r02],
    #      [r10, r11, r12],
    #      [r20, r21, r22]]
    sy = -R[0, 2]
    ry = np.arcsin(sy)

    # Guard for Gimbal lock
    cx =  np.cos(ry)
    if abs(cx) > 1e-6:
        rx = np.arctan2(R[1, 2], R[2, 2])
        rz = np.arctan2(R[0, 1], R[0, 0])
    else:
        # Gimbal lock: fallback
        rx = np.arctan2(-R[2, 1], R[1, 1])
        rz = 0.0

    return np.array([rx, ry, rz], dtype=np.float64)


# ------------------------------------------------------------------------------
# 3. Main function to compute local Euler angles for each of the 25 joints
# ------------------------------------------------------------------------------

def compute_kinect_euler_angles(positions, degrees=False):
    """
    positions: (25, 3) array of XYZ coordinates, indexed by joint.
    degrees: If True, returns angles in degrees; otherwise in radians (default).
    Returns:   (25, 3) array of Euler angles (rx, ry, rz).
    
    The function:
      - Treats SpineBase (joint=0) as the root with an identity orientation
      - Builds each child's absolute rotation matrix based on parent's orientation
      - Converts each child's absolute rotation matrix to local Euler angles (XYZ)
    """
    # Store final Euler angles
    euler_angles = np.zeros((25, 3), dtype=np.float64)

    # This will store a 3x3 absolute orientation matrix for each joint
    absolute_orientations = [np.eye(3) for _ in range(25)]

    # We define a simple reference 'bone direction' for the root: e.g. Y-up
    # (But you could pick something else depending on your coordinate system.)
    root_reference_direction = np.array([0, 1, 0], dtype=np.float64)

    # Traverse in order (naively from 0..24 should work since we have
    # no complicated reordering, but you could also BFS in skeleton tree)
    for joint_idx in range(25):
        parent_idx = PARENTS[joint_idx]

        if parent_idx == -1:
            # Root node (SpineBase)
            # By definition, the root has identity orientation
            absolute_orientations[joint_idx] = np.eye(3)
            euler_angles[joint_idx] = np.zeros(3)
        else:
            parent_pos = positions[parent_idx]
            child_pos  = positions[joint_idx]

            # Vector from parent -> child
            bone_vec = child_pos - parent_pos
            if np.linalg.norm(bone_vec) < 1e-8:
                # Degenerate: child == parent => no orientation
                absolute_orientations[joint_idx] = absolute_orientations[parent_idx]
                euler_angles[joint_idx] = np.zeros(3)
                continue

            # 1) Get parent's absolute orientation
            R_parent = absolute_orientations[parent_idx]

            # 2) Define parent's local "forward" direction in world space
            #    i.e. transform the parent's reference direction out of parent space
            #    into the global (world) space.
            parent_fwd_global = R_parent @ root_reference_direction

            # 3) Compute the rotation that aligns parent's forward vector to bone_vec
            align_rot = rotation_from_vectors(parent_fwd_global, bone_vec)

            # 4) Child's absolute orientation = align_rot * parent's absolute orientation
            #    (Note: matrix multiplication order matters; we are premultiplying
            #     to rotate the parent's orientation in world space.)
            R_child_absolute = align_rot @ R_parent

            # 5) Convert the child's absolute orientation to local Euler angles
            #    relative to the parent's orientation. One way:
            #    local_rotation = R_parent^T * R_child_absolute
            R_local = R_parent.T @ R_child_absolute
            child_euler = rotation_matrix_to_euler_xyz(R_local)

            absolute_orientations[joint_idx] = R_child_absolute
            euler_angles[joint_idx] = child_euler

    # Convert to degrees if requested
    if degrees:
        euler_angles = np.degrees(euler_angles)

    return euler_angles


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    # Load a sample CSV file with 3D joint positions
    df = pd.read_csv("motion_csv_output/S016C001P025R001A020.csv")
    # remove rotation 1-25
    df = df.drop(columns=["rot1_w", "rot1_x", "rot1_y", "rot1_z", "rot2_w", "rot2_x", "rot2_y", "rot2_z", "rot3_w", "rot3_x", "rot3_y", "rot3_z", 
                          "rot4_w", "rot4_x", "rot4_y", "rot4_z", "rot5_w", "rot5_x", "rot5_y", "rot5_z", "rot6_w", "rot6_x", "rot6_y", "rot6_z",
                          "rot7_w", "rot7_x", "rot7_y", "rot7_z", "rot8_w", "rot8_x", "rot8_y", "rot8_z", "rot9_w", "rot9_x", "rot9_y", "rot9_z",
                          "rot10_w", "rot10_x", "rot10_y", "rot10_z", "rot11_w", "rot11_x", "rot11_y", "rot11_z", "rot12_w", "rot12_x", "rot12_y", "rot12_z",
                          "rot13_w", "rot13_x", "rot13_y", "rot13_z", "rot14_w", "rot14_x", "rot14_y", "rot14_z", "rot15_w", "rot15_x", "rot15_y", "rot15_z",
                          "rot16_w", "rot16_x", "rot16_y", "rot16_z", "rot17_w", "rot17_x", "rot17_y", "rot17_z", "rot18_w", "rot18_x", "rot18_y", "rot18_z",
                          "rot19_w", "rot19_x", "rot19_y", "rot19_z", "rot20_w", "rot20_x", "rot20_y", "rot20_z", "rot21_w", "rot21_x", "rot21_y", "rot21_z",
                          "rot22_w", "rot22_x", "rot22_y", "rot22_z", "rot23_w", "rot23_x", "rot23_y", "rot23_z", "rot24_w", "rot24_x", "rot24_y", "rot24_z",
                          "rot25_w", "rot25_x", "rot25_y", "rot25_z"])
    # extract only the position columns
    pos = df.iloc[:, :].values

    print(len(pos), len(pos[0]))
    pos = np.array(pos[0]).reshape(25, 3)

    angles = compute_kinect_euler_angles(pos, degrees=True)
    print("Euler Angles (XYZ in degrees) per joint:\n")
    for i, (rx, ry, rz) in enumerate(angles):
        print(f"Joint {i}: Rx={rx:.1f}, Ry={ry:.1f}, Rz={rz:.1f}")
