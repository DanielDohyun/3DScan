import open3d as o3d
import numpy as np
import os

def segment_cartridge_case(file_path):
    # Load the mesh or point cloud
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".obj":
        mesh = o3d.io.read_triangle_mesh(file_path)
        pcd = mesh.sample_points_poisson_disk(50000)
    elif extension in [".pcd", ".ply"]:
        pcd = o3d.io.read_point_cloud(file_path)
    else:
        print("Unsupported file format!")
        return

    # Visualization 1: Original point cloud
    o3d.visualization.draw_geometries([pcd])

    # Downsample the point cloud
    down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # Plane segmentation
    plane_model, inliers = down_pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1500)
    outlier_cloud = down_pcd.select_by_index(inliers, invert=True)

    # Visualization 2: Outliers after removing the plane (should contain cartridges + other objects)
    o3d.visualization.draw_geometries([outlier_cloud])

# Before clustering the outlier cloud
# increasing the eps value which determines the maximum distance between two samples in the same neighborhood.
# Reducing the min_points value which is the number of samples in a neighborhood for a point to be considered a core point.
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.2, min_points=5, print_progress=True))
    print(f"Unique labels in outlier_cloud: {np.unique(labels)}")

    potential_cartridges = []

    for i in range(labels.max() + 1):
        cluster = outlier_cloud.select_by_index(np.where(labels == i)[0])
        points = np.asarray(cluster.points)
        height_diff = points[:, 2].max() - points[:, 2].min()
        if 1.5 <= height_diff <= 6:  # Loosened height criteria
            potential_cartridges.append(cluster)

    if len(potential_cartridges) < 1:
        print("no img")

    for cartridge in potential_cartridges:
        o3d.visualization.draw_geometries([cartridge])

file_name = "bul2.obj"
segment_cartridge_case(file_name)