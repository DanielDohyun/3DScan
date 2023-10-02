from flask import Flask, render_template, request, send_file
import numpy as np
import cv2
import maskrcnn


# MaskRCNN = maskrcnn.MaskRCNN
app = Flask(__name__, template_folder='.')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded PNG file
        scan_file = request.files['scan']

        # Convert the scan_file object to a string
        scan_filename = scan_file.filename

        # Read the PNG file
        scan = cv2.imread(scan_filename, cv2.IMREAD_GRAYSCALE)

        # Mask the 3D scan using Mask R-CNN
        model = maskrcnn.model()
        model.load_weights("mask_rcnn_cartridge_case.h5")
        results = model.detect([scan])[0]

        # Get the mask of the cartridge case
        mask = results['masks'][:, :, 0]

        # Mask the 3D scan
        masked_scan = scan[mask > 0]

        # Save the masked 3D scan to a file
        cv2.imwrite('masked_scan.png', masked_scan)

        # Send the masked 3D scan to the browser
        return send_file('masked_scan.png', as_attachment=True)

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)