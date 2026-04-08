import zmq
import json
import time
import math

def euler_to_quaternion(roll, pitch, yaw):
    """
    Converts RPY (degrees) to Quaternion [w, x, y, z]
    """
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]

def main():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5556")
    
    print("Target Publisher started on port 5556.")
    print("Format: X, Y, Z, Roll, Pitch, Yaw")
    print("Default: 0.25, 0.25, 1.0, 0, 0, 0")
    print("Press Ctrl+C to exit and hand control back to Isaac Sim GUI.\n")
    
    time.sleep(1)

    try:
        while True:
            user_input = input("Enter X, Y, Z, R, P, Y (comma separated) or Enter for defaults: ").strip()
            
            if user_input == "":
                pos = [0.25, 0.25, 1.0]
                rot_euler = [0.0, 0.0, 0.0]
            else:
                try:
                    vals = [float(i.strip()) for i in user_input.split(",")]
                    if len(vals) == 3:
                        pos = vals
                        rot_euler = [0.0, 0.0, 0.0]
                    elif len(vals) == 6:
                        pos = vals[:3]
                        rot_euler = vals[3:]
                    else:
                        print("Error: Please enter 3 values (pos) or 6 values (pos+rot).")
                        continue
                except ValueError:
                    print("Error: Invalid numerical input.")
                    continue
            
            quat = euler_to_quaternion(rot_euler[0], rot_euler[1], rot_euler[2])
            
            message = {
                "use_zmq_target": True,
                "position": pos,
                "orientation": quat
            }
            
            socket.send_json(message)
            print(f"Sent: Pos={pos}, Rot={rot_euler}")
            print("-" * 30)
            
    except KeyboardInterrupt:
        print("\n\nBroadcasting: DISABLE ZMQ CONTROL")
        socket.send_json({"use_zmq_target": False, "position": [0.25, 0.25, 1.0], "orientation": [1.0, 0, 0, 0]})
        time.sleep(0.5)
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
