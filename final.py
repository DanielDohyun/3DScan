import open3d as o3d
import cv2
import numpy as np
import os

def pcd_image(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(False)
    vis.destroy_window()
    np_img = np.asarray(img)
    return (np_img * 255).astype(np.uint8)

def remove_surface_points(pcd, distance_threshold=0.04):
    #to identify a plane in the point cloud using the RANSAC algorithm.
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return outlier_cloud

def find_cartridge_in_image(image, template_image_path):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    if template_image is None:
        return None

    template_edges = cv2.Canny(template_image, 80, 120)
    # only interested in the contours and not in the hierarchy 
    template_contours, _ = cv2.findContours(template_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    template_contour = max(template_contours, key=cv2.contourArea)
    hu_template = cv2.HuMoments(cv2.moments(template_contour)).flatten()

    # expanding the edges for better contour matching
    kernel = np.ones((3,3), np.uint8)
    image_edges = cv2.Canny(gray_image, 80, 120)
    dilated_image_edges = cv2.dilate(image_edges, kernel, iterations=1)
    image_contours, _ = cv2.findContours(dilated_image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray_image)

    for contour in image_contours:
        hu_contour = cv2.HuMoments(cv2.moments(contour)).flatten()
        delta_hu = np.sum((hu_template - hu_contour)**2)
        if delta_hu < 0.3:  # threshold, adjust if needed
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Morphological closing to clean up the mask
    closing_kernel = np.ones((5,5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
    return cleaned_mask

def filter_point_cloud_by_2d_mask(pcd, mask):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    #Extracting the Dimensions of the Image:
    projected_img_height, projected_img_width = mask.shape
    point_indices_img = -np.ones(mask.shape, dtype=int)

    #determining the bottom-left, top-right corner of the bounding box
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)

    normalized_points = (points - min_bounds) / (max_bounds - min_bounds)

    for i, point in enumerate(normalized_points):
        y = int(point[1] * projected_img_height)
        x = int(point[0] * projected_img_width)
        if 0 <= x < projected_img_width and 0 <= y < projected_img_height:
            point_indices_img[y, x] = i

    #Extracting Point Indices Corresponding to Mask Values:
    valid_point_indices = point_indices_img[mask > 127]
    valid_point_indices = valid_point_indices[valid_point_indices >= 0]
    
    #Extracting the corresponding points from the original points array.
    filtered_points = points[valid_point_indices]
    filtered_colors = colors[valid_point_indices] if colors.size > 0 else np.array([])

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_colors.size > 0:
        new_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return new_pcd

def reconstruct_surface_from_point_cloud(pcd):
    # Estimate the normals
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=40))

    # Use Poisson surface reconstruction to create a mesh
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    return mesh, densities

def trim_mesh(mesh, densities, density_threshold=4.7):
    #for easier filtering
    densities_np = np.asarray(densities)
    vertices_to_remove = densities_np < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh

def smoothen_mesh(mesh, iterations=1):
    for _ in range(iterations):
        mesh = mesh.filter_smooth_simple()
    return mesh

def main(file_path, template_image_path, output_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".obj":
        mesh = o3d.io.read_triangle_mesh(file_path)
        pcd = mesh.sample_points_poisson_disk(50000)
    elif extension in [".pcd", ".ply"]:
        pcd = o3d.io.read_point_cloud(file_path)
    else:
        print("Unsupported file format!")
        return

    o3d.visualization.draw_geometries([pcd])

    # Removing surface points from the 3D point cloud
    pcd_without_surface = remove_surface_points(pcd)
    image = pcd_image(pcd_without_surface)

    mask = find_cartridge_in_image(image, template_image_path)

    if mask is not None:
        cv2.imshow("Cleaned Mask", mask)
        cv2.waitKey(0)
        segmented_pcd = filter_point_cloud_by_2d_mask(pcd_without_surface, mask)

        if not segmented_pcd.has_points():
            print("Segmented point cloud is empty!")
        else:
            # Post-processing: Statistical Outlier Removal
            cl, indices = segmented_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
            cleaned_pcd = segmented_pcd.select_by_index(indices)

            # Calculate the depth directly based on the 3D diameter (0.451 inches)
            # diameter_3d = 0.451
            # max_z = np.max(np.asarray(cleaned_pcd.points)[:, 2])
            # min_z = np.min(np.asarray(cleaned_pcd.points)[:, 2])
            # depth = max_z - min_z

            # Save the segmented point cloud to a PLY file
            mesh, densities = reconstruct_surface_from_point_cloud(cleaned_pcd)
            trimmed_mesh = trim_mesh(mesh, densities)
            smoothed_mesh = smoothen_mesh(trimmed_mesh, iterations=3)

            # Visualize the mesh
            o3d.visualization.draw_geometries([cleaned_pcd])
            o3d.visualization.draw_geometries([smoothed_mesh])

            o3d.io.write_triangle_mesh(output_path, smoothed_mesh)
            print(f"Segmented point cloud saved to {output_path}")

    else:
        print("Mask is None!")
        return

file_name = "bullet1.obj"
template_path = "cCase0.jpeg"
output_path = "segmented_bullet.ply"
main(file_name, template_path, output_path)