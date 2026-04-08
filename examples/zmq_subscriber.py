import zmq
import json
import time

def main():
    # Initialize ZMQ context and SUB socket
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    # Connect to the publisher (default localhost:5555)
    socket.connect("tcp://localhost:5555")
    
    # Subscribe to all messages (empty string prefix)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    print("ZMQ Subscriber started. Waiting for robot joint data...")
    
    try:
        while True:
            # Receive the message
            message = socket.recv_json()
            
            # Print the data
            timestamp = message.get("timestamp", 0)
            joints = message.get("joints", [])
            positions = message.get("positions", [])
            
            print(f"[{timestamp:.3f}] Joint Data Received:")
            for name, pos in zip(joints, positions):
                print(f"  - {name}: {pos:.4f}")
            print("-" * 30)
            
    except KeyboardInterrupt:
        print("\nSubscriber stopped.")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
