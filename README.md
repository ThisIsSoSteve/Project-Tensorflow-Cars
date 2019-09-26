# Project-Tensorflow-Cars

Back again after a lot of learning
Attempting to use Tensorflow to race a car in the game Project Cars. Very much in a work in progress.

## Update
I'm currently rewriting most of the code. I'm simplifying the problem and will be gradually increasing the complexity.

Currently: input: position, angle, velocity of the car, output the acceleration value (0-255)
Next: add braking

I'm not using a conv net yet but planning to at a later stage.

## What I'm using

Windows 10 Pro 64bit

Visual Studio Code https://code.visualstudio.com/

Python 3.7.4 64bit

Project Cars (game) https://www.projectcarsgame.com/

Carseour used for "Project Cars"
pip install git+git://github.com/matslindh/carseour

pypcars2api used for "Project Cars 2" (based on Carseour)
pip install git+git://github.com/marcelomanzo/pypcars2api.git

Tensorflow 2.0.0-rc2 https://www.tensorflow.org/

PYXInput https://github.com/bayangan1991/PYXInput (virtual Xbox controller) 
pip install PYXInput

grabber https://gist.github.com/tzickel/5c2c51ddde7a8f5d87be730046612cd0 (captures the screen)

opencv - image manipulation
pip install opencv-python

console progress bar
pip install tqdm





https://github.com/mhammond/pywin32
pip install pywin32