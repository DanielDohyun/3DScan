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
        depth = (radius**2 - dist_to_axis**2)**0.5
        
        # Add or subtract the depth based on point's position relative to the center
        if np.dot(points[idx] - center, normal) > 0:
            depth = -depth
        depths[idx] = depth

    return depths

def remove_noise_from_point_cloud(pcd, nb_neighbors=60, std_ratio=8.0):
    """
    Remove noise from a point cloud using statistical outlier removal.

    Parameters:
    - pcd: The input point cloud.
    - nb_neighbors: The number of neighbors to analyze for each point.
    - std_ratio: The standard deviation ratio.

    Returns:
    - Cleaned point cloud.
    """
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd



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

    cleaned_pcd = remove_noise_from_point_cloud(new_pcd)

    return cleaned_pcd
    # return new_pcd

def segment_cartridge_case(file_path, template_image_paths):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".obj":
        mesh = o3d.io.read_triangle_mesh(file_path)
        pcd = mesh.sample_points_poisson_disk(50000)
    elif extension in [".pcd", ".ply"]:
        pcd = o3d.io.read_point_cloud(file_path)
        mesh = o3d.io.read_triangle_mesh(file_path)
        o3d.visualization.draw_geometries([mesh], window_name="Original 3D Point Cloud")
        # o3d.visualization.draw_geometries([pcd], window_name="Original 3D Point Cloud")
    else:
        print("Unsupported file format!")
        return

    image = render_point_cloud_to_image(pcd)
    rendered_image = o3d.geometry.Image(image)

    normed_res = find_cartridge_in_image(image, template_image_paths)

    segmented_pcd = geometric_filter(pcd)
    segmented_pcd = filter_point_cloud_by_2d_mask(pcd, normed_res)

    if not segmented_pcd.has_points():
        print("Segmented point cloud is empty!")
    else:
        print(f"Segmented point cloud has {len(segmented_pcd.points)} points.")
        
    o3d.visualization.draw_geometries([segmented_pcd])
    o3d.io.write_point_cloud("output.ply", segmented_pcd)

file_name = "bullet1.ply"
template_images = ["cCase0.jpeg", "cCase2.jpg"]
segment_cartridge_case(file_name, template_images)
#Bullet1 woorks


# import open3d as o3d
# import cv2
# import numpy as np
# import os

# def render_point_cloud_to_image(pcd):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(visible=False)
#     vis.add_geometry(pcd)
#     vis.poll_events()
#     vis.update_renderer()
#     img = vis.capture_screen_float_buffer(False)
#     vis.destroy_window()
#     img_np = np.asarray(img)
#     cv2.imshow("Rendered Image", img_np)
#     cv2.waitKey(3000)
#     return (img_np * 255).astype(np.uint8)

# def find_cartridge_in_image(image):
#     # Display the original image
#     cv2.imshow("Original Image", image)
#     cv2.waitKey(0)

#     if len(image.shape) == 3:
#         image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         image_gray = image

#     # Display the grayscale image
#     cv2.imshow("Grayscale Image", image_gray)
#     cv2.waitKey(0)

#     # Edge detection
#     blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)

#     # Display the edges
#     cv2.imshow("Edges", edges)
#     cv2.waitKey(0)
    
#     # Thresholding
#     thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#     # Display the thresholded image
#     cv2.imshow("Thresholded Image", thresh)
#     cv2.waitKey(0)

#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Sort contours by area and keep the largest few for analysis
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
        
#         # Check if contour matches basic cartridge size characteristics
#         if 0.05 * image_gray.shape[0] <= h <= 0.25 * image_gray.shape[0] and 0.01 * image_gray.shape[1] <= w <= 0.20 * image_gray.shape[1]:
#             mask = np.zeros_like(image_gray)
#             cv2.drawContours(mask, [contour], -1, 255, -1)
#             cv2.imshow("Selected Contour Mask", mask)
#             cv2.waitKey(0)
#             return mask

#     print("No contour matched cartridge characteristics!")
#     return np.zeros_like(image_gray)


# def filter_point_cloud_by_2d_mask(pcd, mask):
#     if len(mask.shape) == 3:
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#     points = np.asarray(pcd.points)
#     colors = np.asarray(pcd.colors)

#     projected_img_height, projected_img_width = mask.shape
#     point_indices_img = -np.ones(mask.shape, dtype=int)

#     for i, point in enumerate(points):
#         y = int(point[1] * projected_img_height)
#         x = int(point[0] * projected_img_width)

#         if 0 <= x < projected_img_width and 0 <= y < projected_img_height:
#             point_indices_img[y, x] = i

#     valid_point_indices = point_indices_img[mask == 255]
#     valid_point_indices = valid_point_indices[valid_point_indices >= 0]

#     filtered_points = points[valid_point_indices]
#     filtered_colors = colors[valid_point_indices] if colors.size > 0 else np.array([])

#     new_pcd = o3d.geometry.PointCloud()
#     new_pcd.points = o3d.utility.Vector3dVector(filtered_points)
#     if filtered_colors.size > 0:
#         new_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

#     return new_pcd


# def segment_cartridge_case(file_path):
#     extension = os.path.splitext(file_path)[1].lower()
#     if extension == ".obj":
#         mesh = o3d.io.read_triangle_mesh(file_path)
#         pcd = mesh.sample_points_poisson_disk(50000)
#     elif extension in [".pcd", ".ply"]:
#         pcd = o3d.io.read_point_cloud(file_path)
#     else:
#         print("Unsupported file format!")
#         return

#     image = render_point_cloud_to_image(pcd)
#     mask = find_cartridge_in_image(image)

#     segmented_pcd = filter_point_cloud_by_2d_mask(pcd, mask)

#     if not segmented_pcd.has_points():
#         print("Segmented point cloud is empty!")
#     else:
#         print(f"Segmented point cloud has {len(segmented_pcd.points)} points.")
        
#     o3d.visualization.draw_geometries([segmented_pcd])

# file_name = "bullet1.obj"
# segment_cartridge_case(file_name)
