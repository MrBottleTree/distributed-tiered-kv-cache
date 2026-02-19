import grpc
import echo_pb2
import echo_pb2_grpc
import threading
import time

def make_messages():
    """A generator that produces messages to send to the server."""
    messages = ["Hello!", "Is anyone there?", "gRPC is cool.", "Goodbye!"]
    for msg in messages:
        time.sleep(1) # Simulated delay between messages
        print(f"[Machine B] Sending: {msg}")
        yield echo_pb2.EchoMessage(name="Machine B (Client)", message=msg)

def run():
    # Connect to the server
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = echo_pb2_grpc.EchoServiceStub(channel)
        
        # Start the bi-directional stream
        responses = stub.Chat(make_messages())
        
        try:
            for response in responses:
                print(f"[Machine B] Received from {response.name}: {response.message}")
        except grpc.RpcError as e:
            print(f"Connection closed: {e}")

if __name__ == '__main__':
    run()
