This document is the user manual of the Imaging Software Development Kit for ST image sensor.
The aim of this document is to provide clarification and example on how to use the IMG_SDK.

It is intended that this document does not cover image sensor configuration to avoid repetition with their respective user manuals.

With the IMG_SDK, you can start an EVK on your own and run it.
The examples provided in this document are based on S5G1 image sensor.
Same logics are applied to other sensors.

## Requirements

- **Software**
  - Windows (10+) or Linux (Ubuntu 20.04+)
  - Python 3.9+
    - python -m pip install -r .\python_requirements.txt
  - Main EVK board drivers
    - Windows [InstallCx3Driver.exe](InstallCx3Driver.exe)
    - Linux [50-cx3.rules](50-cx3.rules)
      ``` cmd
      cp ./50-cx3.rules /etc/udev/rules.d/
      sudo udevadm control --reload
      ```
- **Hardware**
  - Main EVK board
  - USB cable (3.1)

## Package

The SDK package is provided has a zip file with the following name structure:
- ST_BRIGHTSENSE_SDK_SensorName_Major.Minor.BugFix_OS.zip
- Ex: ```ST_BRIGHTSENSE_SDK_VD55G1_1.0.0_windows.zip```

### Files
The package includes different files:
- [cx3_spider_64.dll](cx3_spider_64.dll) / [libcx3_spider_64.so](libcx3_spider_64.so)
  - Library to control the Cx3 (Used by the sensor Library)
- [ST_BRIGHTSENSE_SDK_VD55G1.dll](ST_BRIGHTSENSE_SDK_VD55G1.dll) / [libST_BRIGHTSENSE_SDK_VD55G1.so](libST_BRIGHTSENSE_SDK_VD55G1.so)
  - Sensor library to control the sensor
- [ST_BRIGHTSENSE_SDK_VD55G1_lib.h](IMGST_BRIGHTSENSE_SDK_VD55G1_lib.h)
  - Sensor library header
- [image_sensor_python_sdk.py](image_sensor_python_sdk.py)
  - Python wrapper loading the sensor library
- [smart_imshow.py](smart_imshow.py)
  - Python class to help the live display of images
- [VD55G1_constants.py](VD55G1_constants.py)
  - Aliases to sensor constants as register addresses or LUT
- [VD55G1_example_open_cv.py](VD55G1_example_open_cv.py)
  - Basic example on how to:
    1. Init the board
    2. Set the sensor configuration (Sensor specific)
    3. Start the stream 
    4. Init window for stream 
    5. Loop on frames
    6. Show frames

Other files are other examples of sensor configuration or use cases.