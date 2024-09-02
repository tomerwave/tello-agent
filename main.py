from djitellopy import Tello

# Create a Tello object
tello = Tello()

# Connect to the Tello drone
tello.connect()

# Print battery level
print(f"Battery level: {tello.get_battery()}%")

# Take off
tello.takeoff()

# Move up by 50 cm
tello.move_up(50)

# Move forward by 100 cm
tello.move_forward(50)

# Rotate clockwise by 90 degrees
tello.rotate_clockwise(90)

# Move back by 100 cm
tello.move_back(50)

# Land
tello.land()

# Disconnect
tello.end()
