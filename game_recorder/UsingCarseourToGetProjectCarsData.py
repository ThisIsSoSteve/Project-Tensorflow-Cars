import carseour

# get a live view of the game - this is backed straight from the game's memory, and is updated for each rendered frame
#game = carseour.live()

# get a snapshot of the state of the game - this reads the memory and copies it before returning the object.
#game = carseour.snapshot()

# print current speed of vehicle
#print(game.mSpeed)


import time

game = carseour.live()

while True:
    print('Speed: ', game.mSpeed, 'm/s')
    print('Revs: ', game.mRpm, 'rpm')

    print('Throttle: ', game.mUnfilteredThrottle)#game.mThrottle)
    print('Brakes: ', game.mUnfilteredBrake)#game.mBrake)

    print('Steering: ', game.mUnfilteredSteering)#game.mSteering)

    print('Gear Selected: ', game.mGear)

    print('Wheels Terrain')
    print('[{}][{}]'.format(game.mTerrain[0],game.mTerrain[1]))
    print('[{}][{}]'.format(game.mTerrain[2],game.mTerrain[3]))

    print('Game State', game.mGameState) #GAME_EXITED = 0,GAME_FRONT_END,GAME_INGAME_PLAYING,GAME_INGAME_PAUSED


    #print('Crash State: ', game.mCrashState)
    #print('Brake Damage', game.mBrakeDamage)
    #print('Suspension Damage', game.mSuspensionDamage)
    #print('Aero Damage', game.mAeroDamage)
    #print('Engine Damage', game.mEngineDamage)

    #print('World Position:', game.mWorldPosition)#maybe not needed may over fit to track
    print('Orientation', game.mOrientation[0],game.mOrientation[1],game.mOrientation[2])#[ UNITS = Euler Angles ]???

    #measure how well the car is doing
    #print('LapInvalidated', game.mLapInvalidated)
    #print('BestLapTime', game.mBestLapTime)
    #print('LastLapTime', game.mLastLapTime)
    #print('CurrentTime', game.mCurrentTime)
    #print('CurrentSector1Time', game.mCurrentSector1Time)
    #print('CurrentSector2Time', game.mCurrentSector2Time)
    #print('CurrentSector3Time', game.mCurrentSector3Time)

    #print('mPersonalFastestSector1Time', game.mPersonalFastestSector1Time),
    #print('mPersonalFastestSector2Time', game.mPersonalFastestSector2Time),
    #print('mPersonalFastestSector3Time', game.mPersonalFastestSector3Time),

    ##print('SplitTimeAhead', game.mSplitTimeAhead) #aways -1
    ##print('SplitTimeBehind', game.mSplitTimeBehind) #aways -1
    #print('SplitTime', game.mSplitTime)

    time.sleep(0.5)
