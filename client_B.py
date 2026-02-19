import grpc
import echo_pb2
import echo_pb2_grpc
import threading
import time

# UPDATE THIS IP ADDRESS with the server's actual IP!
SERVER_IP = "10.232.92.100"

def make_messages():
    """Generator to send messages sequentially to the server."""
    messages = ["Hello!", "Is anyone there?", "gRPC is cool.", "Goodbye!"]
    for msg in messages:
        time.sleep(1)  # Delay between messages to simulate real usage
        print(f"[Machine B] Sending: {msg}")
        yield echo_pb2.EchoMessage(name="Machine B (Client)", message=msg)

def run():
    # Attempt to connect to Machine A
    print(f"Connecting to Machine A at {SERVER_IP}:50051...")
    with grpc.insecure_channel(f'{SERVER_IP}:50051') as channel:
        stub = echo_pb2_grpc.EchoServiceStub(channel)
        
        # Open a bi-directional stream
        responses = stub.Chat(make_messages())
        
        try:
            for response in responses:
                print(f"[Machine B] Received from {response.name}: {response.message}")
        except grpc.RpcError as e:
            print(f"Failed to connect or connection lost: {e.code()} - {e.details()}")

if __name__ == '__main__':
    run()
