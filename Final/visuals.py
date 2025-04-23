from PIL import Image

# List of the saved plot files
image_files = ['C:/Users/wasay/Desktop/DSC 540/Final/dataPic.png', 
               'C:/Users/wasay/Desktop/DSC 540/Final/dataNoCurricularPic.png', 
               'C:/Users/wasay/Desktop/DSC 540/Final/featureEngineeringPic.png']

# Open all images
images = [Image.open(img) for img in image_files]

max_width = max(img.width for img in images)
total_height = sum(img.height for img in images)

# Create a new blank image to stitch all the images together (vertically)
stitched_image = Image.new('RGB', (max_width, total_height))

# Paste each image into the new image, one on top of the other
y_offset = 0
for img in images:
    stitched_image.paste(img, (0, y_offset))  # Paste at the current y_offset
    y_offset += img.height  # Update y_offset for the next image

# Save the stitched image
stitched_image.save('stitched_roc_curve_vertical.png')

# Show the stitched image
stitched_image.show()