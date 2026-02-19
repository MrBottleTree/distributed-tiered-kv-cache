import grpc
import echo_pb2
import echo_pb2_grpc
from concurrent import futures

class EchoServiceServicer(echo_pb2_grpc.EchoServiceServicer):
    def Chat(self, request_iterator, context):
        """
        Processes messages sent from the client and echoes them back.
        """
        for request in request_iterator:
            print(f"[Machine A] Received from {request.name}: {request.message}")
            
            # Echo response back to the client
            yield echo_pb2.EchoMessage(
                name="Machine A (Server)", 
                message=f"Echoing: {request.message}"
            )

def serve():
    # Setup the gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    echo_pb2_grpc.add_EchoServiceServicer_to_server(EchoServiceServicer(), server)
    
    # Listen on all interfaces ([::]) at port 50051
    server.add_insecure_port('[::]:50051')
    print("Machine A (Server) is running and waiting for a connection on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
