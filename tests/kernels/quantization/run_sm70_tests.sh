#!/bin/bash
#
# Build and run SM70 Marlin MMA comprehensive test suite
#
# Usage:
#   ./run_sm70_tests.sh                  # Build and run all tests
#   ./run_sm70_tests.sh build            # Build only
#   ./run_sm70_tests.sh run              # Run only (assumes built)
#   ./run_sm70_tests.sh correctness      # Run specific section
#   ./run_sm70_tests.sh benchmark        # Run benchmarks
#   ./run_sm70_tests.sh variant          # Run variant tests
#   ./run_sm70_tests.sh summary          # Show design summary
#   ./run_sm70_tests.sh clean            # Remove build artifacts
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_FILE="test_marlin_mma_sm70.cu"
BINARY="test_marlin_mma_sm70"
INCLUDE_DIR="../../../csrc/quantization/marlin"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

cd "$SCRIPT_DIR"

build() {
    echo -e "${YELLOW}Building SM70 MMA test suite...${NC}"
    
    # Check for CUDA
    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}Error: nvcc not found. Please ensure CUDA is installed.${NC}"
        exit 1
    fi
    
    # Compile
    echo "  nvcc -o $BINARY $SOURCE_FILE -I$INCLUDE_DIR -arch=sm_70 -O3"
    nvcc -o "$BINARY" "$SOURCE_FILE" -I"$INCLUDE_DIR" -arch=sm_70 -O3
    
    echo -e "${GREEN}Build successful!${NC}"
}

run() {
    if [ ! -f "$BINARY" ]; then
        echo -e "${RED}Error: Binary not found. Run './run_sm70_tests.sh build' first.${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Running SM70 MMA test suite...${NC}"
    ./"$BINARY" "$@"
}

clean() {
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    rm -f "$BINARY"
    echo -e "${GREEN}Clean complete.${NC}"
}

# Main
case "${1:-all}" in
    build)
        build
        ;;
    run)
        shift
        run "$@"
        ;;
    clean)
        clean
        ;;
    correctness|numerical|stress|benchmark|variant|summary)
        if [ ! -f "$BINARY" ]; then
            build
        fi
        run "$1"
        ;;
    all|"")
        build
        run
        ;;
    help|--help|-h)
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  build         Build the test binary"
        echo "  run           Run all tests"
        echo "  clean         Remove build artifacts"
        echo "  correctness   Run correctness tests only"
        echo "  numerical     Run numerical edge case tests"
        echo "  stress        Run stress tests"
        echo "  benchmark     Run performance benchmarks"
        echo "  variant       Run variant function tests"
        echo "  summary       Show design decision summary"
        echo "  all           Build and run all tests (default)"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Run '$0 --help' for usage."
        exit 1
        ;;
esac
