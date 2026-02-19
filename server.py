import grpc
import echo_pb2
import echo_pb2_grpc
from concurrent import futures
import time

class EchoServiceServicer(echo_pb2_grpc.EchoServiceServicer):
    def Chat(self, request_iterator, context):
        """
        request_iterator is a generator that yields messages from the client.
        The server can yield (send back) responses one-by-one as it receives them.
        """
        for request in request_iterator:
            print(f"[Machine A] Received from {request.name}: {request.message}")
            
            # This is the "Echo" back to the client
            yield echo_pb2.EchoMessage(
                name="Machine A (Server)", 
                message=f"Echoing: {request.message}"
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    echo_pb2_grpc.add_EchoServiceServicer_to_server(EchoServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("Machine A (Server) started on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
