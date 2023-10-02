import open3d as o3d
import numpy as np
import os
import cv2

def load_3d_file(filename):
    print("Loading 3D Data...")
    if filename.endswith(".obj"):
        pcd = o3d.io.read_triangle_mesh(filename)
    elif filename.endswith(".ply"):
        pcd = o3d.io.read_point_cloud(filename)
    else:
        raise ValueError("Unsupported file format")
    
    o3d.visualization.draw_geometries([pcd], window_name='Loaded 3D Data')
    return pcd

def project_3d_to_2d(pcd):
    print("Projecting to 2D...")
    if isinstance(pcd, o3d.geometry.TriangleMesh):
        pcd = pcd.sample_points_uniformly(number_of_points=50000)
    points = np.asarray(pcd.points)
    
    xy = points[:, :2] - np.min(points[:, :2], axis=0)
    xy = (xy / np.max(xy, axis=0) * 255).astype(np.int)
    
    img = np.zeros((256, 256), dtype=np.uint8)
    for x, y in xy:
        img[y, x] += 1

    img = (img / np.max(img) * 255).astype(np.uint8)
    
    o3d.visualization.draw_geometries([o3d.geometry.Image(img)], window_name='2D Projection')
    return img

def apply_edge_detection(img):
    print("Applying Edge Detection...")
    edges = cv2.Canny(img, 50, 150)
    o3d.visualization.draw_geometries([o3d.geometry.Image(edges)], window_name='Edge Detection')
    return edges

def region_growing_3d(pcd):
    print("Applying Region Growing on 3D Data...")
    
    # Convert mesh to point cloud if necessary
    if isinstance(pcd, o3d.geometry.TriangleMesh):
        pcd = pcd.sample_points_uniformly(number_of_points=50000)
    
    # Assuming the cartridge case is the largest cluster, segment using DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
    max_label = labels.max()
    
    # Get the largest cluster
    largest_cluster = pcd.select_by_index(np.where(labels == max_label)[0])
    
    o3d.visualization.draw_geometries([largest_cluster], window_name='Region Growing 3D')
    return largest_cluster

def save_3d_file(pcd, filename):
    print(f"Saving to {filename}...")
    if filename.endswith(".obj"):
        o3d.io.write_triangle_mesh(filename, pcd)
    elif filename.endswith(".ply"):
        o3d.io.write_point_cloud(filename, pcd)
    else:
        raise ValueError("Unsupported file format")

def process_3d_file(input_filename):
    basename = os.path.basename(input_filename)
    output_filename = os.path.join(os.getcwd(), "processed_" + basename)

    pcd = load_3d_file(input_filename)
    img = project_3d_to_2d(pcd)
    apply_edge_detection(img)
    filtered_pcd = region_growing_3d(pcd)
    save_3d_file(filtered_pcd, output_filename)

filename = "bullet1.obj"  # or .ply
process_3d_file(filename)
