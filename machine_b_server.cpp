#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <memory>

#include <grpcpp/grpcpp.h>
#include "cpp_server/kv_cache.grpc.pb.h"

using namespace std;
using namespace grpc;
using namespace kvcache::v1;
using namespace filesystem;

// implementation of the actual dummy external KV Cache
class KVCacheServerImpl : public KVCacheService::Service {
private:
    const string CACHE_ROOT = "cache";

    string resolve_block_path(const string& session_id, int64_t sequence_index) {
        string filename = session_id + "_seq_" + to_string(sequence_index) + ".pb";
        return CACHE_ROOT + "/" + filename;
    }

public:
    KVCacheServerImpl() {
        if (!exists(CACHE_ROOT)) {
            create_directories(CACHE_ROOT);
        }
        cout << "[Server] Initialized. Storage: ./" << CACHE_ROOT << endl;
    }

    // Stores a block sent from Machine A
    Status Store(ServerContext* context, const KVCacheBlock* block, StoreResponse* response) override {

        cout<<"GOT SOME DATA!"<<endl;
        cout<<"Session ID: "<<block->metadata().session_id()<<endl;
        
        string path = resolve_block_path(
            block->metadata().session_id(),
            block->metadata().sequence_index()
        );

        ofstream output_file(path, ios::binary);
        if (!output_file.is_open()) {
            response->set_success(false);
            response->set_message("Failed to open file: " + path);
            return Status(StatusCode::INTERNAL, "File I/O error");
        }

        if (block->SerializeToOstream(&output_file)) {
            response->set_success(true);
            response->set_message("Stored.");
            return Status::OK;
        } else {
            response->set_success(false);
            response->set_message("Failed to serialize.");
            return Status(StatusCode::INTERNAL, "Protobuf error");
        }
    }

    // Returns a block to Machine A
    Status Fetch(ServerContext* context, const ContextSignal* signal, KVCacheBlock* response_block) override {

        string path = resolve_block_path(signal->session_id(), signal->request_sequence_id());

        ifstream input_file(path, ios::binary);
        if (!input_file.is_open()) {
            return Status(StatusCode::NOT_FOUND, "Block not found: " + path);
        }

        if (response_block->ParseFromIstream(&input_file)) {
            return Status::OK;
        } else {
            return Status(StatusCode::INTERNAL, "Failed to parse from disk.");
        }
    }

    // Handles live updates and proactive pushes
    Status StreamContext(ServerContext* context, ServerReaderWriter<KVCacheBlock, ContextSignal>* stream) override {
        
        ContextSignal signal;
        cout << "[Stream] Connected." << endl;

        while (stream->Read(&signal)) {
            string session_id = signal.session_id();

            if (signal.has_query_vector()) {
                cout << "[Stream] Semantic query: " << session_id << endl;
                // Search logic goes here
            } 
            else if (signal.request_sequence_id() != 0) {
                int64_t seq_id = signal.request_sequence_id();
                string path = resolve_block_path(session_id, seq_id);

                ifstream input_file(path, ios::binary);
                KVCacheBlock block;

                if (input_file.is_open() && block.ParseFromIstream(&input_file)) {
                    cout << "[Stream] Pushing " << seq_id << endl;
                    stream->Write(block);
                }
            }
        }

        cout << "[Stream] Disconnected." << endl;
        return Status::OK;
    }
};

void RunServer(string port) {
    string address = "0.0.0.0:" + port;
    KVCacheServerImpl service;

    ServerBuilder builder;
    // Increase max message size to 128MB using both methods for compatibility
    builder.SetMaxReceiveMessageSize(128 * 1024 * 1024);
    builder.SetMaxSendMessageSize(128 * 1024 * 1024);
    builder.AddChannelArgument(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, 128 * 1024 * 1024);
    builder.AddChannelArgument(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, 128 * 1024 * 1024);

    builder.AddListeningPort(address, InsecureServerCredentials());
    builder.RegisterService(&service);

    unique_ptr<Server> server = builder.BuildAndStart();
    cout << "[Server] Listening on " << address << endl;
    server->Wait();
}

int main(int argc, char* argv[]) {
    RunServer(argv[1]);
    return 0;
}
