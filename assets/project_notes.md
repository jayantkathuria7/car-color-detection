### 🛠️ Project Customizations as per Task Requirements:

This app includes specific customizations that were implemented according to the instructions given by the project supervisor. Please keep these in mind when interpreting the results:

- **Car color prediction customization**:
    - Predictions for **red** and **blue** cars have been **intentionally swapped**.
    - This was a **task-specific instruction** and does **not reflect a bug or model error**.
    - For example: if the model predicts `red`, the app will show `blue`, and vice versa.

- **Object detection filter**:
    - Only objects with bounding boxes larger than `80x80 px` are considered to avoid small, unclear detections.

- **Face detection and age/gender prediction**:
    - Age and gender detection is only triggered when **more than one face** is detected.
    - Gender prediction is binary (`Male` / `Female`) based on the model design.

- **Vehicle classification**:
    - Vehicle types are detected using the COCO SSD Mobilenet model.
    - Categories include `car`, `bus`, `motorcycle`, `truck`, etc., based on COCO labels.

> ⚠️ Note: These customizations were made intentionally for demonstration and evaluation purposes. Accuracy should be interpreted in the context of these task-specific changes.
"""
