import json
import os
import random
from typing import List, Dict
import cv2
import inquirer
import numpy as np
import tyro
from tyro.conf import Positional
from dataclasses import dataclass
from typing import Literal, Optional
from typing_extensions import Annotated
from pathlib import Path

available_intrinsics = {
    'Custom': {
        "camera_model": "OPENCV",  # Camera model
        "fl_x": 320.0,  # focal length x
        "fl_y": 320.0,  # focal length y
        "cx": 320.0,  # principal point x
        "cy": 240.0,  # principal point y
        "w": 640,  # image width
        "h": 480  # image height
    },

    'Air': {
        "camera_model": "OPENCV", # Camera model
        "fl_x": 320.0, # focal length x
        "fl_y": 320.0, # focal length y
        "cx": 320.0, # principal point x
        "cy": 240.0, # principal point y
        "w": 640, # image width
        "h": 480 # image height
    },
    'Ground': {
        "camera_model": "OPENCV", # Camera model
        "fl_x": 320.0, # focal length x
        "fl_y": 320.0, # focal length y
        "cx": 320.0, # principal point x
        "cy": 320.0, # principal point y
        "w": 640, # image width
        "h": 640 # image height
    }
}

@dataclass
class TartanToNerfStudio:
    """Converter for TartanAir/Ground dataset to NerfStudio format."""
    
    base_path: Positional[Path]
    """Path to the base folder of the TartanAir/Ground dataset."""

    pose_limit: Annotated[Optional[int], tyro.conf.arg(aliases=["-p"])] = None
    """Limit the number of poses to convert."""
    uniform: Annotated[bool, tyro.conf.arg(aliases=["-u"])] = False
    """If True, distribute the poses uniformly."""
    spatial_uniform: Annotated[bool, tyro.conf.arg(aliases=["-su"])] = False
    """If True, distribute poses uniformly in 3D space using spatial cubes."""
    cube_size: Annotated[float, tyro.conf.arg(aliases=["-cs"])] = 10.0
    """Size of cubes for spatial uniform distribution (in meters)."""
    depth_conversion: Annotated[bool, tyro.conf.arg(aliases=["-d"])] = False
    """Convert depth images to npy files."""
    camera_intrinsics: Annotated[Literal["Air", "Ground", "Custom"], tyro.conf.arg(aliases=["-c"])] = "Air"
    """Camera intrinsics to use. (Air, Ground, Custom)"""
    output_path: Annotated[Optional[Path], tyro.conf.arg(aliases=["-o"])] = None
    """Path to the output transforms.json file. If None, it will be saved in the base path."""
    
    def validate_args(self):
        if self.camera_intrinsics not in ["Air", "Ground", "Custom"]:
            raise ValueError("Camera intrinsics must be either 'Air', 'Ground' or 'Custom'.")
        else:
            print(f"Using camera intrinsics: Tartan-{self.camera_intrinsics}")

        if self.spatial_uniform and self.uniform:
            raise ValueError("Cannot use both 'uniform' and 'spatial_uniform' at the same time. Please choose one.")

        if self.output_path is None:
            self.output_path = self.base_path / "transforms.json"

    def check_poses(self):
        """
        Check which poses are available in the dataset.
        :return: List of available poses.
        """

        poses = os.listdir(self.base_path)
        poses = [p for p in poses if p.startswith("pose") and p.endswith(".txt")]

        pose_info = []

        for pose in poses:
            # Check if pose has image and/or depth images
            name = pose.replace("pose", "").replace(".txt", "")

            has_rgb = (self.base_path / f"image{name}").exists()
            has_depth = (self.base_path / f"depth{name}").exists()

            if not has_rgb:
                continue

            pose_info.append({
                "name": name,
                "has_rgb": has_rgb,
                "has_depth": has_depth
            })

        return pose_info

    def spatial_uniform_distribution(self, all_camera_poses: List[List[float]], limit: int) -> List[int]:
        """
        Select camera indices using spatial uniform distribution based on 3D cube subdivision.
        
        :param all_camera_poses: List of all camera poses [x, y, z, qx, qy, qz, qw]
        :param limit: Maximum number of cameras to select
        :return: List of selected camera indices
        """
        if len(all_camera_poses) <= limit:
            return list(range(len(all_camera_poses)))
        
        # Extract camera positions (x, y, z)
        positions = np.array([[pose[0], pose[1], pose[2]] for pose in all_camera_poses])
        
        # Calculate scene bounds
        min_bounds = positions.min(axis=0)
        max_bounds = positions.max(axis=0)
        
        # Subdivide scene into cubes
        # Calculate number of cubes in each dimension
        num_cubes = np.ceil((max_bounds - min_bounds) / self.cube_size).astype(int)
        num_cubes = np.maximum(num_cubes, 1)  # At least 1 cube per dimension
        
        print(f"Scene bounds: min={min_bounds}, max={max_bounds}")
        print(f"Subdividing into {num_cubes[0]}x{num_cubes[1]}x{num_cubes[2]} cubes (cube size: {self.cube_size}m)")
        
        # Assign each camera to a cube
        cube_indices = {}
        for idx, pos in enumerate(positions):
            # Calculate which cube this position belongs to
            cube_coords = np.floor((pos - min_bounds) / self.cube_size).astype(int)
            cube_coords = np.clip(cube_coords, 0, num_cubes - 1)
            cube_key = tuple(cube_coords)
            
            if cube_key not in cube_indices:
                cube_indices[cube_key] = []
            cube_indices[cube_key].append(idx)
        
        # Filter to only cubes with at least one camera
        cubes_with_cameras = {k: v for k, v in cube_indices.items() if len(v) > 0}
        num_cubes_with_cameras = len(cubes_with_cameras)
        
        print(f"Found {num_cubes_with_cameras} cubes with cameras (out of {np.prod(num_cubes)} total cubes)")
        
        if num_cubes_with_cameras == 0:
            # Fallback: just take first N cameras
            return list(range(min(limit, len(all_camera_poses))))
        
        # Calculate target cameras per cube
        target_per_cube = limit // num_cubes_with_cameras
        remainder = limit % num_cubes_with_cameras
        
        # Sort cubes by number of cameras (ascending)
        sorted_cubes = sorted(cubes_with_cameras.items(), key=lambda x: len(x[1]))
        
        selected_indices = []
        
        # First pass: assign target_per_cube cameras to each cube
        for cube_key, camera_indices in sorted_cubes:
            num_cameras_in_cube = len(camera_indices)
            num_to_select = min(target_per_cube, num_cameras_in_cube)
            
            if num_to_select > 0:
                # Uniformly sample from this cube
                if num_to_select == num_cameras_in_cube:
                    # Take all cameras
                    selected_indices.extend(camera_indices)
                else:
                    # Uniformly sample
                    step = num_cameras_in_cube / num_to_select
                    sampled = [camera_indices[int(i * step)] for i in range(num_to_select)]
                    selected_indices.extend(sampled)
        
        # Second pass: distribute remainder starting from cubes with fewest cameras
        if remainder > 0:
            for cube_key, camera_indices in sorted_cubes:
                if remainder <= 0:
                    break
                
                # Count how many we already selected from this cube
                already_selected = [idx for idx in selected_indices if idx in camera_indices]
                num_available = len(camera_indices) - len(already_selected)
                
                if num_available > 0:
                    # Select one more from this cube, uniformly randomly from remaining
                    remaining_indices = [idx for idx in camera_indices if idx not in already_selected]
                    if len(remaining_indices) > 0:
                        # Randomly select from remaining indices
                        selected_indices.append(random.choice(remaining_indices))
                        remainder -= 1
        
        # Sort indices to maintain original order
        selected_indices = sorted(selected_indices)
        
        print(f"Selected {len(selected_indices)} cameras from {num_cubes_with_cameras} cubes")
        
        return selected_indices

    def create_transforms(self, poses: List[str]):
        """
        Create the transforms.json file for the NerfStudio dataset.
        :param poses: List of poses to include in the dataset.
        :return: None
        """

        loaded_poses = self.load_poses(poses)
        converted_poses = self.convert_poses(loaded_poses)
        self.write_transforms(converted_poses)

    def load_poses(self, poses: List[str]):
        """
        Load the poses from the dataset.
        :param poses: List of poses to load.
        :return: List of loaded poses.
        """
        loaded_poses = []
        
        # First, load all poses without limiting (for spatial uniform distribution)
        all_poses_data = []
        for name in poses:
            with open(self.base_path / f"pose{name}.txt") as f:
                lines = f.readlines()
                # Separate the values and convert them to float
                camera_poses = [[float(x) for x in line.strip().split()] for line in lines]
                has_depth = (self.base_path / f"depth{name}").exists()
                
                all_poses_data.append({
                    'name': name,
                    'camera_poses': camera_poses,
                    "has_depth": has_depth,
                    'start_idx': sum(len(p['camera_poses']) for p in all_poses_data),
                    'end_idx': sum(len(p['camera_poses']) for p in all_poses_data) + len(camera_poses)
                })
        
        # Apply spatial uniform distribution if enabled
        if self.pose_limit is not None and self.spatial_uniform:
            # Collect all camera poses
            all_camera_poses = []
            for pose_data in all_poses_data:
                all_camera_poses.extend(pose_data['camera_poses'])
            
            # Apply spatial uniform distribution
            selected_indices = self.spatial_uniform_distribution(all_camera_poses, self.pose_limit)
            
            # Map selected indices back to pose files
            for pose_data in all_poses_data:
                # Find indices that belong to this pose file
                pose_indices = [idx for idx in selected_indices 
                              if pose_data['start_idx'] <= idx < pose_data['end_idx']]
                # Convert to local indices within this pose file
                local_indices = [idx - pose_data['start_idx'] for idx in pose_indices]
                local_indices = sorted(local_indices)  # Keep sorted for consistency
                # Select the corresponding camera poses
                pose_data['camera_poses'] = [pose_data['camera_poses'][i] for i in local_indices]
                pose_data['selected_indices'] = local_indices  # Store for use in write_transforms
                print(f"Selected {len(pose_data['camera_poses'])} poses from {pose_data['name']}")
        
        # Apply regular uniform or simple limit distribution
        elif self.pose_limit is not None:
            for pose_data in all_poses_data:
                single_limit = self.pose_limit // len(poses)
                print(f"Limiting poses to {single_limit} per pose file (total: {len(pose_data['camera_poses'])})")
                if self.uniform:
                    distribution = len(pose_data['camera_poses']) // single_limit
                    print(f"Distributing poses uniformly with a step of {distribution} (total: {len(pose_data['camera_poses'])})")
                    pose_data['camera_poses'] = pose_data['camera_poses'][::distribution]
                else:
                    pose_data['camera_poses'] = pose_data['camera_poses'][:single_limit]
        
        # Create final structure
        for pose_data in all_poses_data:
            loaded_pose = {
                'name': pose_data['name'],
                'camera_poses': pose_data['camera_poses'],
                "has_depth": pose_data['has_depth']
            }
            # Store selected indices if spatial uniform was used
            if 'selected_indices' in pose_data:
                loaded_pose['selected_indices'] = pose_data['selected_indices']
            loaded_poses.append(loaded_pose)

        return loaded_poses

    def convert_poses(self, poses: List[Dict]):
        """
        Convert the poses to NerfStudio format.
        :param poses: List of poses to convert.
        :return: List of converted poses.
        """
        for pose in poses:
            pose['nerfstudio_poses'] = self.convert_pose(pose)

        return poses

    def convert_pose(self, pose: Dict):
        nerfstudio_poses = []
        for camera_pose in pose['camera_poses']:
            x_ned, y_ned, z_ned, qx_ned, qy_ned, qz_ned, qw_ned = camera_pose
            transform_ned = self.quaternion_to_transform(x_ned, y_ned, z_ned, qx_ned, qy_ned, qz_ned, qw_ned)
            nerfstudio_poses.append(self.ned_to_opengl(transform_ned))

        return nerfstudio_poses

    @staticmethod
    def quaternion_to_transform(x: float,
                                y: float,
                                z: float,
                                qx: float,
                                qy: float,
                                qz: float,
                                qw: float) -> np.ndarray:
        """
        Converts OpenGL-compatible coordinates and quaternion into a 4x4 transformation matrix.

        :param x: OpenGL X coordinate (translation)
        :param y: OpenGL Y coordinate (translation)
        :param z: OpenGL Z coordinate (translation)
        :param qx: Quaternion i
        :param qy: Quaternion j
        :param qz: Quaternion k
        :param qw: Quaternion scalar (real part)
        :return: 4x4 transformation matrix in OpenGL coordinates.
        """
        # Normalize the quaternion to ensure it's valid
        norm = np.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

        # Compute the rotation matrix from the quaternion
        rotation_matrix = np.array([
            [qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2, 2 * (qx * qy - qw * qz), 2 * (qw * qy + qx * qz)],
            [2 * (qx * qy + qw * qz), qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2, 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qw * qx + qy * qz), qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2]
        ])

        # Construct the 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix  # Set the rotation matrix
        transform_matrix[:3, 3] = [x, y, z]  # Set the translation

        return transform_matrix

    @staticmethod
    def ned_to_opengl(transform_ned: np.ndarray) -> np.ndarray:
        """
        Converts a 4x4 transformation matrix from NED to OpenGL coordinates.

        :param transform: 4x4 transformation matrix in NED coordinates.
        :return: 4x4 transformation matrix in OpenGL
        """
        # NED to OpenGL transformation matrix
        translation = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        translation_inv = np.linalg.inv(translation)

        transform_opengl = translation @ transform_ned @ translation_inv

        return transform_opengl

    def convert_depth_image(self, depth_path):
        """
        For TartanGround 32bit depth images are saved as PNG files. This function converts them to npy files.
        :param depth_path:
        :return:
        """
        depth_rgba = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        # Convert depth image to float32 format
        depth_values = depth_rgba.view("<f4").squeeze()

        # Save to npy file
        converted_path = str(depth_path).replace(".png", "_converted.npy")
        np.save(converted_path, depth_values)

        return os.path.basename(converted_path)

    def write_transforms(self, poses):
        """
        Write the transforms.json file.
        :param poses: List of poses to write.
        :return:
        """
        global available_intrinsics

        transforms = available_intrinsics[self.camera_intrinsics].copy()
        transforms["frames"] = []

        for pose in poses:
            image_folder_name = f"image{pose['name']}"
            depth_folder_name = f"depth{pose['name']}"

            images_folder = self.base_path / image_folder_name
            images = [i for i in os.listdir(images_folder) if i.endswith((".png", ".jpg", ".jpeg"))]
            images = sorted(images, key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))

            if self.pose_limit is not None:
                if self.spatial_uniform and 'selected_indices' in pose:
                    # Use the same indices that were selected for poses
                    images = [images[i] for i in pose['selected_indices']]
                else:
                    single_limit = self.pose_limit // len(poses)
                    if self.uniform:
                        distribution = len(images) // single_limit
                        images = images[::distribution]
                    else:
                        images = images[:single_limit]

            if len(images) != len(pose['nerfstudio_poses']):
                raise ValueError(f"Number of images ({len(images)}) does not match the number of poses ({len(pose['nerfstudio_poses'])})")

            if pose['has_depth']:
                depth_folder = self.base_path / depth_folder_name
                depth_images = [i for i in os.listdir(depth_folder) if (i.endswith((".npy", ".png", ".jpg", ".jpeg")) and not i.endswith("_converted.npy"))]
                depth_images = sorted(depth_images, key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))

                if self.pose_limit is not None:
                    if self.spatial_uniform and 'selected_indices' in pose:
                        # Use the same indices that were selected for poses
                        depth_images = [depth_images[i] for i in pose['selected_indices']]
                    else:
                        single_limit = self.pose_limit // len(poses)
                        if self.uniform:
                            distribution = len(depth_images) // single_limit
                            depth_images = depth_images[::distribution]
                        else:
                            depth_images = depth_images[:single_limit]

                if len(depth_images) != len(pose['nerfstudio_poses']):
                    raise ValueError(f"Number of depth images ({len(depth_images)}) does not match the number of poses ({len(pose['nerfstudio_poses'])})")

            for i, nerf_pose in enumerate(pose['nerfstudio_poses']):
                frame = {
                    "file_path": f"{image_folder_name}/{images[i]}",
                    "transform_matrix": nerf_pose.tolist()
                }

                if pose['has_depth']:
                    depth_file = f"{depth_folder_name}/{depth_images[i]}"
                    if self.depth_conversion:
                        converted_file = self.convert_depth_image(self.base_path / depth_file)
                        frame["depth_file_path"] = f"{depth_folder_name}/{converted_file}"
                    else:
                        frame["depth_file_path"] = depth_file

                transforms["frames"].append(frame)

        with open(self.output_path, "w") as f:
            json.dump(transforms, f, indent=4)
            print(f'Transforms file written to {self.output_path}')
    

def entrypoint():
    """
    Entrypoint for the script.
    :return: 
    """

    tyro.extras.set_accent_color('bright_yellow')
    converter = tyro.cli(TartanToNerfStudio)
    converter.validate_args()
    poses = converter.check_poses()
    questions = [
        inquirer.Checkbox('poses',
                          message='Select which poses you want to have in the dataset.',
                          choices=[(f"{pose['name'].strip('_')} (rgb{'+d' if pose['has_rgb'] else ''})", pose['name'])
                                   for pose in poses],
                          ),
    ]
    converter.create_transforms(inquirer.prompt(questions)['poses'])


if __name__ == "__main__":
    entrypoint()