from flask import Flask, render_template, request, send_file
import numpy as np
import cv2

def mask_scan(scan, threshold=0.5):
  """Masks a 3D scan using a simple thresholding algorithm.

  Args:
    scan: A NumPy array containing the 3D scan.
    threshold: The threshold value. Pixels above the threshold will be masked.

  Returns:
    A NumPy array containing the masked 3D scan.
  """

  # Check if the scan is valid
  if scan is None:
    raise ValueError("Scan is None")

  # Mask the 3D scan
  mask = scan > threshold

  # Apply the mask to the 3D scan
  masked_scan = scan[mask]

  return masked_scan

app = Flask(__name__, template_folder='.')
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        scan_file = request.files['scan']

        # Check if the uploaded image file is a valid PNG file
        if not scan_file.filename.endswith('.png'):
            return render_template('error.html', error="Invalid image file format")

        # Read the uploaded image file
        scan = cv2.imread(scan_file, cv2.IMREAD_GRAYSCALE)

        # Check if the cv2.imread() function was able to read the image file
        if scan is None:
            return render_template('error.html', error="Failed to read image file")

        # Mask the 3D scan
        masked_scan = mask_scan(scan)

        # Save the masked 3D scan to a file
        cv2.imwrite('masked_scan.png', masked_scan)

        # Send the masked 3D scan to the browser
        return send_file('masked_scan.png', as_attachment=True)

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)