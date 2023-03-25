# emotion-detection
A program that can detect emotion from video and imagery

## Description
The goal of this project is to be able to detect emotion from video provided by a face camera. Long term we would like to integrate with an object detectin model like yolo or maybe a facial recognition algorithm. Then use that detection to create a chip that we can send to our emotion processor to detect the emotion and return it to the original program.

## Getting Started
- Make sure you have Python 3.10 (3.10.10)
    - I recommend creating a conda environment specific to this project
    - `echo y | conda create --name emotion-detection python=3.10.10`
- Install the required python packages
    - `pip install -r requirements.txt`

- to test on an image you can run the following
    - `python image_test.py`
    - NOTE: there are settings you can adjust at the top of the file like if you want to add a different image

- to test on a webcam you can run the following
    - `python webcam_test.py`
    - NOTE: there are settings you can adjust at the top of the file like if you want to add a different video capture device
## TODO
- If face is neutral maybe give the second highest confidence emotion as well
- Maybe show all emotions in order with confidence greater than an adjustable value (0.25, 0.1)
- Make boxes and labels fancier
- Make emotion detection faster/more efficient