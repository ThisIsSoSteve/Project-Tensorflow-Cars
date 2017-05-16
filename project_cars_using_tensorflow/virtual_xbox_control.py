import pyxinput
import numpy as np
#pyxinput.test_virtual()
MyVirtual = pyxinput.vController()

control_throttle = 'TriggerR'
control_brakes = 'TriggerL'
control_steering = 'AxisLx'


def control_car(throttle, brakes, steering_left, steering_right):

    #throttle = np.absolute(throttle)
    #brakes = np.absolute(brakes)
    #steering_left = np.absolute(steering_left)
    #steering_right = np.absolute(steering_right)

    if throttle > 0.9:
        throttle = 1
    if brakes > 0.9:
        brakes = 1
    if steering_left > 0.05:
        steering_left = 1
        steering_right = 0
    if steering_right > 0.05:
        steering_right = 1
        steering_left = 0

    #if steering_left < 0.1:
    #    steering_left = 0
    #if steering_right < 0.1:
    #    steering_right = 0
    
    #if steering_left > steering_right:
    #    MyVirtual.set_value(control_steering, steering_left)
    #else:
    #    MyVirtual.set_value(control_steering, steering_right)

    steering_left = -steering_left
    MyVirtual.set_value(control_steering, steering_left + steering_right)

     
    if throttle > brakes:
        MyVirtual.set_value(control_throttle, throttle)
        MyVirtual.set_value(control_brakes, 0)
    else:
        MyVirtual.set_value(control_brakes, brakes)
        MyVirtual.set_value(control_throttle, 0)
    
    print("Throttle:", throttle, "Brakes:", brakes, "Steering:", steering_left + steering_right)

'''Set a value on the controller
    All controls will accept a value between -1.0 and 1.0
    Control List:
    AxisLx          , Left Stick X-Axis
    AxisLy          , Left Stick Y-Axis
    AxisRx          , Right Stick X-Axis
    AxisRy          , Right Stick Y-Axis
    BtnBack         , Menu/Back Button
    BtnStart        , Start Button
    BtnA            , A Button
    BtnB            , B Button
    BtnX            , X Button
    BtnY            , Y Button
    BtnThumbL       , Left Thumbstick Click
    BtnThumbR       , Right Thumbstick Click
    BtnShoulderL    , Left Shoulder Button
    BtnShoulderR    , Right Shoulder Button
    Dpad            , Set Dpad Value (0 = Off, Use DPAD_### Contstants)
    TriggerL        , Left Trigger
    TriggerR        , Right Trigger
'''