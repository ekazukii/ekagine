# EXE is the full output path (like gcc -o)
EXE ?= ./app

# Build and copy
all:
	@echo "Building release..."
	@BIN_PATH=$$(cargo build --release --message-format=json \
		| jq -r 'select(.executable) | .executable') && \
	echo "Copying $$BIN_PATH -> $(EXE)" && \
	cp "$$BIN_PATH" "$(EXE)"

clean:
	cargo clean
