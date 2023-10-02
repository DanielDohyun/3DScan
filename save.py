import open3d as o3d
import cv2
import numpy as np
import os

def render_point_cloud_to_image(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(False)
    vis.destroy_window()
    img_np = np.asarray(img)
    return (img_np * 255).astype(np.uint8)
    
def geometric_filter(pcd):
    bbox = pcd.get_axis_aligned_bounding_box()
    extents = bbox.get_extent()

    min_length, max_length = 2, 5  # in cm
    min_width, max_width = 0.8, 1.5  # in cm

    # Checking if the extents fit within the range for a cartridge case
    if min_length <= extents[2] <= max_length and min_width <= extents[0] <= max_width:
        return pcd
    else:
        return o3d.geometry.PointCloud()  # return an empty point cloud

def find_cartridge_in_image(image, templates):
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    blurred_image_gray = cv2.GaussianBlur(image_gray, (7, 7), 0)
    
    max_normed_res = None

    for idx, template_path in enumerate(templates):
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        if template.shape[0] > image_gray.shape[0] or template.shape[1] > image_gray.shape[1]:
            template = cv2.resize(template, (image_gray.shape[1] - 1, image_gray.shape[0] - 1))

        blurred_template = cv2.GaussianBlur(template, (7, 7), 0)
        res = cv2.matchTemplate(image_gray, blurred_template, cv2.TM_CCOEFF_NORMED)
        # res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
        
        normed_res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # cv2.imshow("Normalized Template Matching Result - " + template_path, normed_res)
        # cv2.waitKey(0)

        if idx == 0:
            max_normed_res = normed_res
        else:
            if max_normed_res.shape == normed_res.shape:
                max_normed_res = cv2.max(normed_res, max_normed_res)

    # cv2.imshow("Maximum Normalized Result Across Templates", max_normed_res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return max_normed_res


def fit_cylinder_to_points(points):
    """
    Fit a cylinder to a set of points using PCA. 
    Return center, normal, and radius of the fitted cylinder.
    """
    mean = np.mean(points, axis=0)
    
    # PCA to get the principal components
    cov_matrix = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # The smallest eigenvalue corresponds to the normal direction
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    
    # Project points onto plane with normal direction
    projected_points = points - mean
    distances = np.dot(projected_points, normal)
    
    # Distance to the furthest point will give an approximation of the radius
    radius = np.max(np.abs(distances))

    return mean, normal, radius


def estimate_depths_from_cylinder(points, center, normal, radius):
    """
    Use the cylindrical nature of the cartridge to estimate the depth values for points.
    """
    depths = np.zeros(points.shape[0])

    # Calculate depth for each point
    for idx in range(points.shape[0]):
        point_on_axis = center + np.dot(points[idx] - center, normal) * normal
        dist_to_axis = np.linalg.norm(points[idx] - point_on_axis)
        
        # Calculate depth using Pythagoras' theorem
        depth = max(0, radius**2 - dist_to_axis**2)**0.5

        # Add or subtract the depth based on point's position relative to the center
        if np.dot(points[idx] - center, normal) > 0:
            depth = -depth
        depths[idx] = depth

    return depths

def filter_point_cloud_by_2d_mask(pcd, normed_res):
    threshold = 0.7 * 255
    valid_mask = np.where(normed_res > threshold, 255, 0)

    kernel = np.ones((5,5), np.uint8)
    valid_mask = cv2.dilate(valid_mask.astype(np.uint8), kernel, iterations=1)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Determine 3D bounds of point cloud
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)

    # Normalize points to [0, 1] range
    normalized_points = (points - min_bounds) / (max_bounds - min_bounds)

    projected_img_height, projected_img_width = normed_res.shape
    point_indices_img = -np.ones(normed_res.shape, dtype=int)

    for i, point in enumerate(normalized_points):
        # Convert normalized points to image coordinates
        y = int(point[1] * projected_img_height)
        x = int(point[0] * projected_img_width)

        if 0 <= x < projected_img_width and 0 <= y < projected_img_height:
            point_indices_img[y, x] = i

    # Get the valid points based on the mask
    valid_point_indices_from_img = point_indices_img[valid_mask == 255]
    valid_point_indices_from_img = valid_point_indices_from_img[valid_point_indices_from_img >= 0]
    valid_points_for_cylinder = points[valid_point_indices_from_img]
    
    # Fit a cylinder to these valid points
    center, normal, radius = fit_cylinder_to_points(valid_points_for_cylinder)

# Estimate the depth values using the cylinder fitting
    depths_for_valid_points = estimate_depths_from_cylinder(valid_points_for_cylinder, center, normal, radius)

# Use these depths for the valid points
    valid_points_for_cylinder[:, 2] = depths_for_valid_points


    # Replace the depths of the valid points in the original point cloud
    points[valid_point_indices_from_img] = valid_points_for_cylinder

    # Create a new point cloud with the updated points and colors and return it
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(points)
    if colors.size > 0:
        new_pcd.colors = o3d.utility.Vector3dVector(colors[valid_point_indices_from_img])

    cleaned_pcd = remove_outliers_combined(new_pcd)

    return cleaned_pcd
    # return new_pcd

def remove_outliers_combined(pcd, nb_neighbors=40, std_ratio=7.0, nb_points=15, radius=0.05):
    """
    Apply a combined outlier removal technique.
    """
    # Statistical Outlier Removal
    cleaned_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    # Radius Outlier Removal
    cleaned_pcd, _ = cleaned_pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return cleaned_pcd

def smooth_mesh(mesh, iterations=10, lambda_value=0.5):
    """
    Apply the Laplacian smoothing to the mesh.
    """
    for i in range(iterations):
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=1, lambda_filter=lambda_value)
    return mesh

def reconstruct_surface_from_point_cloud(pcd):
    """
    Reconstruct the surface mesh from point cloud using Poisson reconstruction.
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]

def segment_cartridge_case(file_name, template_images):
    pcd = o3d.io.read_point_cloud(file_name)
    image = render_point_cloud_to_image(pcd)

    normed_res = find_cartridge_in_image(image, template_images)
    segmented_pcd = filter_point_cloud_by_2d_mask(pcd, normed_res)
    o3d.visualization.draw_geometries([segmented_pcd])
    o3d.io.write_point_cloud("segmented_bullet.ply", segmented_pcd)

    # Reconstructing the surface from the point cloud
    reconstructed_mesh = reconstruct_surface_from_point_cloud(segmented_pcd)
    # Smoothing the mesh
    smoothed_mesh = smooth_mesh(reconstructed_mesh)
    # Visualizing the smoothed mesh
    o3d.visualization.draw_geometries([smoothed_mesh])

# main entry point
file_name = "bullet1.ply"
template_images = ["cCase0.jpeg", "cCase2.jpg"]
segment_cartridge_case(file_name, template_images)