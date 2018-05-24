import pyxinput
import numpy as np

class virtual_xbox_controller:

    def __init__(self):
         self.MyVirtual = pyxinput.vController()
         self.control_throttle = 'TriggerR'
         self.control_brakes = 'TriggerL'
         self.control_steering = 'AxisLx'
         self.steering = 0.0

    def control_car(self, throttle, brakes, steering_left, steering_right):

        if throttle > 0.95:
            throttle = 1.0
        if brakes > 0.6:
            throttle  = 0
        if brakes > 0.8:
            brakes = 1.0
            throttle  = 0
        if steering_left > 0.9:
            steering_left = 0.4
            steering_right = 0
        if steering_right > 0.9:
            steering_right = 0.4
            steering_left = 0

        #if steering_left < 0.05:
        #    steering_left = 0
        #if steering_right < 0.05:
        #    steering_right = 0

    
        #if steering_left > steering_right:
        #    MyVirtual.set_value(control_steering, steering_left)
        #else:
        #    MyVirtual.set_value(control_steering, steering_right)
        #steering_left = steering_left * 10
        #steering_right = steering_right * 10

        #if steering_left < 0.1:
        #    steering_left = 0
        #if steering_right < 0.1:
        #    steering_right = 0

        steering_left = -steering_left
        self.steering = steering_left + steering_right

        self.MyVirtual.set_value(self.control_steering, steering_left + steering_right)

        #self.steering += (-steering_left + steering_right)

        #self.steering = max(min(1, self.steering), -1) #clamp

        #self.MyVirtual.set_value(self.control_steering, self.steering)
     
        if throttle > brakes:
            self.MyVirtual.set_value(self.control_throttle, throttle - brakes)
            self.MyVirtual.set_value(self.control_brakes, 0)
            brakes = 0
        else:
            self.MyVirtual.set_value(self.control_brakes, brakes)
            self.MyVirtual.set_value(self.control_throttle, 0)
            throttle = 0
    
        print("Throttle:", throttle, "Brakes:", brakes, "Steering:", self.steering)

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