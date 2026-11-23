Auto-Find

Auto-Find detects people and selected items in a video, tracks them with
YOLOv8m and ByteTrack, and saves evidence when an item is marked abandoned.

Folder layout

  project/main.py                Video processing and alert logic
  project/custom_bytetrack.yaml  ByteTrack configuration
  project/yolov8m.pt             Model weights
  project/sample2.mp4            Sample video
  project/requirements.txt       Python dependencies
  docs/Auto-Find Pseudocode.txt   Report-ready algorithm descriptions

Run the sample on Windows

From PowerShell opened in the Auto-Find folder, run these commands with
Python 3.12 installed:

  cd project
  py -3.12 -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  .\.venv\Scripts\python.exe main.py sample2.mp4

Run the script from the project folder so it can find the model and tracker
configuration. To use another video, pass its path in place of sample2.mp4.

Press Space to pause or resume, Q to quit, and use the timeline to seek.
The application creates project/alerts when an alert is triggered. It stores
JPEG crops and JSON metadata there.

The active abandonment threshold is 10 warning frames. The startup message
reports a five-second threshold calculated from the video frame rate.
