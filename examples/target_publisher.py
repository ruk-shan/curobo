import zmq
import json
import time

def main():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    # Port 5556 for target and control commands
    socket.bind("tcp://*:5556")
    
    print("Target Publisher started on port 5556.")
    print("Default settings: position=[0.25, 0.25, 1.0], orientation=[1.0, 0, 0, 0], use_zmq_target=True")
    print("Press Ctrl+C to exit and hand control back to Isaac Sim GUI.\n")
    
    # Wait a moment for the subscriber to potentially connect
    time.sleep(1)

    try:
        while True:
            # Prompt user for input
            user_input = input("Enter cube position X, Y, Z (comma separated) or press Enter for default: ").strip()
            
            if user_input == "":
                # Default settings
                pos = [0.25, 0.25, 1.0]
            else:
                try:
                    pos = [float(i.strip()) for i in user_input.split(",")]
                    if len(pos) != 3:
                        print("Error: Please enter exactly 3 values for X, Y, Z.")
                        continue
                except ValueError:
                    print("Error: Invalid input. Please enter numbers separated by commas.")
                    continue
            
            # Construct message
            message = {
                "use_zmq_target": True,
                "position": pos,
                "orientation": [1.0, 0.0, 0.0, 0.0]  # Default neutral orientation
            }
            
            # Broadcast the target
            socket.send_json(message)
            print(f"Sent Target: {pos} (remote_control=True)")
            
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
