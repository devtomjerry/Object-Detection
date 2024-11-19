from arcgis.gis import GIS
from arcgis.learn import Model
from PIL import Image
import matplotlib.pyplot as plt

# Skip logging into ArcGIS Online (Optional)
gis = GIS()  # This won't require arcpy

# Load the image (JPEG image, not a raster dataset)
image_data_path = "./car.jpg"
image = Image.open(image_data_path)

# Display the image (optional, for verification)
plt.imshow(image)
plt.axis("off")
plt.show()

# Load the pretrained car detection model from the .dlpk file
model_path = r"C:\Users\Jay Sharma\Desktop\Object Detection\CarDetection_USA.dlpk"
car_detection_model = Model.from_model(model_path)

# Run prediction on the image
predictions = car_detection_model.predict(image)

# Save predictions (bounding boxes, labels, etc.)
output_path = "./output/predicted_car_locations"
predictions.save(output_path)

# Optionally, display the results
predictions.show_results()

# from arcgis.learn import MaskRCNN  # Correct import for object detection
# from arcgis.gis import GIS
# from PIL import Image
# import matplotlib.pyplot as plt

# # Initialize GIS (you may or may not need to log in)
# gis = GIS()

# # Load the image
# image_data_path = "./car.jpg"
# image = Image.open(image_data_path)

# # Display the image
# plt.imshow(image)
# plt.axis("off")
# plt.show()

# # Load the model
# model_path = "./CarDetection_USA.dlpk"
# car_detection_model = MaskRCNN.from_model(model_path)  # Correct way to load the model

# # Run prediction
# predictions = car_detection_model.predict(image)

# # Save the predictions (bounding boxes, labels, etc.)
# output_path = "./output/predicted_car_locations"
# predictions.save(output_path)

# # Show results (optional)
# predictions.show_results()
