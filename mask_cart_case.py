# import cv2
# import numpy as np
# import os

# class CartridgeCaseMasker:
#     def __init__(self, google_images):
#         self.google_images = google_images
#         self.detector = cv2.CascadeClassifier("frontalface_default.xml")

#     def mask_cartridge_case(self, image):
#         # Convert the image to grayscale.
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Detect the cartridge case in the image.
#         detections = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#         # If no cartridge case is detected, return the original image.
#         if len(detections) == 0:
#             return image

#         # Create a mask for the cartridge case.
#         mask = np.zeros(image.shape[:2], dtype=np.uint8)
#         for (x, y, w, h) in detections:
#             cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

#         # Apply the mask to the image.
#         masked_image = cv2.bitwise_and(image, image, mask=mask)

#         return masked_image

# def main():
#     # Load the local images.
#     google_images = []
#     for image_name in ["cartCase1.jpeg", "cartCase2.jpeg"]:
#         image_path = os.path.join(os.getcwd(), image_name)
#         google_images.append(cv2.imread(image_path))

#     # Create a CartridgeCaseMasker object.
#     cartridge_case_masker = CartridgeCaseMasker(google_images)

#     # Load the 3D scan of the cartridge case.
#     cartridge_case_scan = cv2.imread("cartridge_case_scan.jpg")

#     # Mask the 3D scan of the cartridge case.
#     masked_cartridge_case_scan = cartridge_case_masker.mask_cartridge_case(cartridge_case_scan)

#     # Save the masked 3D scan of the cartridge case.
#     cv2.imwrite("masked_cartridge_case_scan.jpg", masked_cartridge_case_scan)

# if __name__ == "__main__":
#     main()


import trimesh

class CartridgeCaseMasker:
    def __init__(self):
        pass

    def mask_cartridge_case(self, mesh):
        # Sample a point cloud from the mesh.
        points, _ = trimesh.sample.sample_surface_even(mesh, 10000)

        # Create a point cloud from the sampled points.
        point_cloud = trimesh.PointCloud(points)

        # Fit a plane to the point cloud.
        plane = point_cloud.plane_fit_pca()

        # Create a mask for the cartridge case.
        mask = trimesh.boolean(mesh.vertices, plane, invert=True)

        # Apply the mask to the mesh.
        masked_mesh = trimesh.Trimesh(mesh.vertices[mask], mesh.faces.copy())

        return masked_mesh

def main():
    # Open the 3D scan of the cartridge case.
    with open("cart.ply", "rb") as file_obj:
        mesh = trimesh.load(file_obj, file_type="ply")

    # Create a CartridgeCaseMasker object.
    cartridge_case_masker = CartridgeCaseMasker()

    # Mask the 3D scan of the cartridge case.
    masked_mesh = cartridge_case_masker.mask_cartridge_case(mesh)

    # Save the masked 3D scan of the cartridge case.
    trimesh.io.export(masked_mesh, "masked_cart.ply")

if __name__ == "__main__":
    main()
