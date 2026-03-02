.PHONY: setup run clean logs proto_py proto_cpp server_cpp

PROTO_SRC = proto/kv_cache.proto
PY_OUT = ./py_client
CPP_OUT = ./cpp_server

proto_py:
	@echo "Generating Python gRPC stubs..."
	mkdir -p $(PY_OUT)
	uv run python -m grpc_tools.protoc -I./proto --python_out=$(PY_OUT) --grpc_python_out=$(PY_OUT) $(PROTO_SRC)
	touch $(PY_OUT)/__init__.py

proto_cpp:
	@echo "Generating C++ gRPC stubs..."
	mkdir -p $(CPP_OUT)
	protoc -I./proto --cpp_out=$(CPP_OUT) --grpc_out=$(CPP_OUT) --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` $(PROTO_SRC)

compile_cpp: proto_cpp
	@echo "Compiling C++ Machine B server..."
	g++ -std=c++17 machine_b_server.cpp $(CPP_OUT)/*.pb.cc -o machine_b_server `pkg-config --cflags --libs protobuf grpc++` -lpthread -ldl

setup:
	@echo "Syncing dependencies with uv..."
	uv sync
	@echo "Ensuring cache directory exists..."
	mkdir -p ./hf_cache
	@if [ ! -f .env ]; then cp .env.example .env; echo "\nWARNING: Created .env file. Open it and add your HF_TOKEN!"; fi

run:
	@echo "Starting vLLM server..."
	uv run --env-file .env vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
	  --dtype auto \
	  --max-model-len 8192 \
	  --gpu-memory-utilization 0.90 \
	  --enforce-eager

clean_cache:
	@echo "Wiping Hugging Face cache..."
	rm -rf ./hf_cache/*

logs:
	@echo "Tailing AWS startup logs..."
	sudo tail -f /var/log/cloud-init-output.log

run_quantized:
	@echo "Starting Official Meta Llama 3.1 8B..."
	uv run --env-file .env vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
		--dtype bfloat16 \
		--max-model-len 20000 \
		--gpu-memory-utilization 0.90 \
		--enforce-eager
