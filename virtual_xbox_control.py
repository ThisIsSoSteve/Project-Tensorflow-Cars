import pyxinput
import numpy as np

class virtual_xbox_controller:

    def __init__(self):
         self.MyVirtual = pyxinput.vController()
         self.control_throttle = 'TriggerR'
         self.control_brakes = 'TriggerL'
         self.control_steering = 'AxisLx'
         #self.steering = 0.0

    def control_car(self, throttle_brake, steering_left_right):

        throttle = 0.0
        brake = 0.0
        
        if throttle_brake > 0.0:
            brake = 0.0
            throttle = throttle_brake
        else:
            throttle = 0.0
            
            brake = throttle_brake * -1

        if throttle > 0.91:
            throttle = 1.0

        if brake > 0.90:
            brake = 1.0

        if throttle < 0.01:
            throttle = 0.0

        if brake < 0.01:
            brake = 0.0

        if steering_left_right > 0.95:
            steering_left_right = 1.0

        
        if steering_left_right < -0.95:
            steering_left_right = -1.0


        # if steering_left_right > -0.03 and steering_left_right < 0.03:
        #     steering_left_right = 0.0


        self.MyVirtual.set_value(self.control_steering, steering_left_right)
        self.MyVirtual.set_value(self.control_throttle, throttle)#/ 255.0
        self.MyVirtual.set_value(self.control_brakes, brake)

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

        # steering_left = -steering_left
        # self.steering = steering_left + steering_right

        # self.MyVirtual.set_value(self.control_steering, steering_left + steering_right)

        #self.steering += (-steering_left + steering_right)

        #self.steering = max(min(1, self.steering), -1) #clamp

        #self.MyVirtual.set_value(self.control_steering, self.steering)
     
        # if throttle > brakes:
        #     self.MyVirtual.set_value(self.control_throttle, throttle - brakes)
        #     self.MyVirtual.set_value(self.control_brakes, 0)
        #     brakes = 0
        # else:
        #     self.MyVirtual.set_value(self.control_brakes, brakes)
        #     self.MyVirtual.set_value(self.control_throttle, 0)
        #     throttle = 0
    
        print('Throttle: {}, Brakes: {}, Steering: {}'.format(throttle, brake, steering_left_right))

    def control_car_throttle_only(self, action):
        throttle = 0.0
        brake = 0.0
        
        if action > 0:
            brake = 0
            throttle = action
        else:
            throttle = 0
            brake = action * -1

        if throttle > 0.97:
            throttle = 1.0

        if brake > 0.97:
            brake = 1.0

        if throttle < 0.1:
            throttle = 0

        if brake < 0.1:
            brake = 0
            
        self.MyVirtual.set_value(self.control_throttle, throttle)#/ 255.0
        self.MyVirtual.set_value(self.control_brakes, brake)

        print('Throttle: {}, Brakes: {}'.format(throttle, brake))


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