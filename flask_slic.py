from flask import Flask, render_template, request, send_file
import open3d as o3d
from skimage.segmentation import slic, mark_boundaries

app = Flask(__name__)

def mask_scan(scan_array):
  """Masks a 3D scan.

  Args:
    scan_array: A NumPy array containing the 3D scan.

  Returns:
    A NumPy array containing the masked 3D scan.
  """

  # Perform SLIC segmentation on the 3D scan
  segments = slic(scan_array, n_segments=200)

  # Mark the boundaries of the segments
  boundaries = mark_boundaries(segments)

  # Create a mask of the cartridge case
  mask = np.zeros_like(scan_array)
  mask[boundaries > 0] = 1

  # Mask the 3D scan
  masked_scan = scan_array[mask > 0]

  return masked_scan

@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    # Get the uploaded PNG file
    scan_file = request.files['scan']

    # Load the PNG file
    scan_array = o3d.io.read_image(scan_file)

    # Mask the 3D scan
    masked_scan = mask_scan(scan_array)

    # Save the masked 3D scan to a file
    o3d.io.write_point_cloud('masked_scan.pcd', o3d.geometry.PointCloud(masked_scan))

    # Send the masked 3D scan to the browser
    return send_file('masked_scan.pcd', as_attachment=True)

  else:
    return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
