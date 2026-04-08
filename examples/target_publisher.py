import zmq
import json
import time
import math

def euler_to_quaternion(roll, pitch, yaw):
    """
    Converts RPY (degrees) to Quaternion [w, x, y, z]
    """
    # Convert degrees to radians
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)

    # Calculate half angles
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Compute quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]

def main():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    # Port 5556 for target and control commands
    socket.bind("tcp://*:5556")
    
    print("Target Publisher started on port 5556.")
    print("Default settings: position=[0.25, 0.25, 1.0], rotation=[0,0,0], use_zmq_target=True")
    print("Press Ctrl+C to exit and hand control back to Isaac Sim GUI.\n")
    
    # Wait a moment for the subscriber to potentially connect
    time.sleep(1)

    try:
        while True:
            # 1. Prompt for position
            pos_input = input("Enter position X, Y, Z (comma separated) or Enter for [0.25, 0.25, 1.0]: ").strip()
            if pos_input == "":
                pos = [0.25, 0.25, 1.0]
            else:
                try:
                    pos = [float(i.strip()) for i in pos_input.split(",")]
                    if len(pos) != 3:
                        print("Error: Please enter exactly 3 values for X, Y, Z.")
                        continue
                except ValueError:
                    print("Error: Invalid position input.")
                    continue

            # 2. Prompt for rotation
            rot_input = input("Enter rotation Roll, Pitch, Yaw (degrees, comma separated) or Enter for [0, 0, 0]: ").strip()
            if rot_input == "":
                rot_euler = [0.0, 0.0, 0.0]
            else:
                try:
                    rot_euler = [float(i.strip()) for i in rot_input.split(",")]
                    if len(rot_euler) != 3:
                        print("Error: Please enter exactly 3 values for R, P, Y.")
                        continue
                except ValueError:
                    print("Error: Invalid rotation input.")
                    continue
            
            # Convert Euler to Quaternion
            quat = euler_to_quaternion(rot_euler[0], rot_euler[1], rot_euler[2])
            
            # Construct message
            message = {
                "use_zmq_target": True,
                "position": pos,
                "orientation": quat
            }
            
            # Broadcast the target
            socket.send_json(message)
            print(f"Sent Target: Pos={pos}, Rot={rot_euler} (remote_control=True)")
            print("-" * 30)
            
    except KeyboardInterrupt:
        # Hand control back to Isaac Sim on exit
        print("\n\nBroadcasting: DISABLE ZMQ CONTROL")
        socket.send_json({
            "use_zmq_target": False,
            "position": [0.25, 0.25, 1.0],
            "orientation": [1.0, 0.0, 0.0, 0.0]
        })
        time.sleep(0.5)
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
