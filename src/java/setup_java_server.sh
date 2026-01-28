#!/bin/bash
# Setup script for Py4J Java Parser Server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAVA_DIR="$SCRIPT_DIR"
BUILD_DIR="$JAVA_DIR/build"
TEMP_DIR="/tmp/astParser_extracted"

echo "Setting up Java Parser Server for Py4J..."

# Check if Java is installed
if ! command -v javac &> /dev/null; then
    echo "Error: Java compiler (javac) not found. Please install Java JDK."
    exit 1
fi

if ! command -v java &> /dev/null; then
    echo "Error: Java runtime (java) not found. Please install Java JDK."
    exit 1
fi

# Check Java version (should be 8+)
JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | sed '/^1\./s///' | cut -d'.' -f1)
if [ "$JAVA_VERSION" -lt 8 ]; then
    echo "Warning: Java 8 or higher is recommended. Found version: $JAVA_VERSION"
fi

# Extract the Java source files
echo "Extracting Java source files..."
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
cd "$JAVA_DIR"

if [ -f "astParser.zip" ]; then
    unzip -q -o astParser.zip -d "$TEMP_DIR"
    SOURCE_DIR="$TEMP_DIR/me/asif/astparser"
elif [ -f "astParserJAVA.zip" ]; then
    unzip -q -o astParserJAVA.zip -d "$TEMP_DIR"
    SOURCE_DIR="$TEMP_DIR/me/asif/astparser"
else
    echo "Error: No Java source archive found (astParser.zip or astParserJAVA.zip)"
    exit 1
fi

# Create build directory
echo "Creating build directory..."
mkdir -p "$BUILD_DIR"

# Check if Py4J JAR is available
PY4J_JAR=$(python3 -c "import py4j; import os; print(os.path.join(os.path.dirname(py4j.__file__), 'java_gateway.jar'))" 2>/dev/null || echo "")

if [ -z "$PY4J_JAR" ] || [ ! -f "$PY4J_JAR" ]; then
    echo "Warning: Py4J JAR not found. Trying to locate it..."
    # Try common locations
    if [ -f "$(pip show py4j | grep Location | cut -d' ' -f2)/py4j/java_gateway.jar" ]; then
        PY4J_JAR="$(pip show py4j | grep Location | cut -d' ' -f2)/py4j/java_gateway.jar"
    else
        echo "Error: Py4J JAR not found. Please ensure py4j is installed: pip install py4j"
        exit 1
    fi
fi

echo "Found Py4J JAR: $PY4J_JAR"

# Compile Java files
echo "Compiling Java source files..."
cd "$SOURCE_DIR"
javac -cp "$PY4J_JAR" -d "$BUILD_DIR" *.java

if [ $? -ne 0 ]; then
    echo "Error: Java compilation failed"
    exit 1
fi

echo "✓ Java files compiled successfully"

# Create a run script
cat > "$JAVA_DIR/start_java_server.sh" << 'EOF'
#!/bin/bash
# Start the Py4J Java Parser Server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
PY4J_JAR=$(python3 -c "import py4j; import os; print(os.path.join(os.path.dirname(py4j.__file__), 'java_gateway.jar'))" 2>/dev/null)

if [ ! -f "$PY4J_JAR" ]; then
    PY4J_JAR="$(pip show py4j | grep Location | cut -d' ' -f2)/py4j/java_gateway.jar"
fi

echo "Starting Py4J Java Parser Server..."
echo "Py4J JAR: $PY4J_JAR"
echo "Build directory: $BUILD_DIR"
echo ""
echo "Server will listen on port 25333 (default Py4J port)"
echo "Press Ctrl+C to stop the server"
echo ""

cd "$BUILD_DIR"
java -cp ".:$PY4J_JAR" me.asif.astparser.AstEntryPoint
EOF

chmod +x "$JAVA_DIR/start_java_server.sh"

echo ""
echo "✓ Setup complete!"
echo ""
echo "To start the Java server, run:"
echo "  $JAVA_DIR/start_java_server.sh"
echo ""
echo "Or run it in the background:"
echo "  nohup $JAVA_DIR/start_java_server.sh > /tmp/java_server.log 2>&1 &"
echo ""
echo "The server will listen on port 25333 (default Py4J port)"
