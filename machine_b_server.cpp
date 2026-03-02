#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <memory>

#include <grpcpp/grpcpp.h>
#include "cpp_server/kv_cache.grpc.pb.h"

using namespace kvcache::v1;
namespace fs = std::filesystem;

/**
 * Machine B: Smart Memory Server implementation.
 * This server manages the KV Cache blocks by persisting them to disk.
 */
class KVCacheServerImpl final : public KVCacheService::Service {
    private:
        const std::string CACHE_ROOT = "cache";

        /**
        * Helper to generate a unique file path for a specific cache block.
        * Format: cache/{session_id}_seq_{sequence_index}.pb
        */
        std::string resolve_block_path(const std::string& session_id, int64_t sequence_index) {
            return CACHE_ROOT + "/" + session_id + "_seq_" + std::to_string(sequence_index) + ".pb";
        }

    public:
        KVCacheServerImpl() {
            if (!fs::exists(CACHE_ROOT)) {
                fs::create_directories(CACHE_ROOT);
            }
            std::cout << "[Server] Initialized. Storage directory: ./" << CACHE_ROOT << std::endl;
        }

        /**
        * Unary RPC: Receives a KV block from Machine A and writes it to disk.
        */
        grpc::Status Store(grpc::ServerContext* context, 
                        const KVCacheBlock* block, 
                        StoreResponse* response) override {
            
            std::string path = resolve_block_path(block->metadata().session_id(), 
                                                block->metadata().sequence_index());

            std::ofstream output_file(path, std::ios::binary);
            if (!output_file.is_open()) {
                response->set_success(false);
                response->set_message("Failed to open file for writing: " + path);
                return grpc::Status(grpc::StatusCode::INTERNAL, "File I/O error");
            }

            if (block->SerializeToOstream(&output_file)) {
                response->set_success(true);
                response->set_message("Block stored successfully.");
                return grpc::Status::OK;
            } else {
                response->set_success(false);
                response->set_message("Serialization failed.");
                return grpc::Status(grpc::StatusCode::INTERNAL, "Protobuf serialization error");
            }
        }

        /**
        * Unary RPC: Retrieves a specific KV block from disk and returns it to Machine A.
        */
        grpc::Status Fetch(grpc::ServerContext* context, 
                        const ContextSignal* signal, 
                        KVCacheBlock* response_block) override {

            std::string path = resolve_block_path(signal->session_id(), 
                                                signal->request_sequence_id());

            std::ifstream input_file(path, std::ios::binary);
            if (!input_file.is_open()) {
                return grpc::Status(grpc::StatusCode::NOT_FOUND, "Cache block not found at: " + path);
            }

            if (response_block->ParseFromIstream(&input_file)) {
                return grpc::Status::OK;
            } else {
                return grpc::Status(grpc::StatusCode::INTERNAL, "Failed to parse cache block from disk.");
            }
        }

        /**
        * Bidirectional Stream: Receives signals (queries/hints) and pushes blocks back proactively.
        */
        grpc::Status StreamContext(grpc::ServerContext* context, 
                                grpc::ServerReaderWriter<KVCacheBlock, ContextSignal>* stream) override {
            
            ContextSignal incoming_signal;
            std::cout << "[Stream] New connection from Machine A." << std::endl;

            while (stream->Read(&incoming_signal)) {
                std::string session_id = incoming_signal.session_id();

                // Case 1: Machine A is performing a semantic search (Query Vector)
                if (incoming_signal.has_query_vector()) {
                    std::cout << "[Stream] Processing semantic query for session: " << session_id << std::endl;
                    // Future: Integrate FAISS search here.
                } 
                
                // Case 2: Machine A explicitly requested a block sequence
                else if (incoming_signal.request_sequence_id() != 0) {
                    int64_t seq_id = incoming_signal.request_sequence_id();
                    std::string path = resolve_block_path(session_id, seq_id);

                    std::ifstream input_file(path, std::ios::binary);
                    KVCacheBlock block_to_push;

                    if (input_file.is_open() && block_to_push.ParseFromIstream(&input_file)) {
                        std::cout << "[Stream] Pushing block " << seq_id << " to Machine A." << std::endl;
                        stream->Write(block_to_push);
                    }
                }
            }

            std::cout << "[Stream] Connection closed." << std::endl;
            return grpc::Status::OK;
        }
};

/**
 * Entry point to start the Machine B server.
 */
void RunServer() {
    std::string server_address("0.0.0.0:50051");
    KVCacheServerImpl service;

    grpc::ServerBuilder builder;
    // Listen on port 50051 without SSL/TLS for this prototype.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "[Server] Machine B (C++) listening on " << server_address << std::endl;
    server->wait_for_termination();
}

int main(int argc, char** argv) {
    RunServer();
    return 0;
}
