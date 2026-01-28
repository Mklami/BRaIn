# Java Parser Server Setup

This directory contains the Java source code for the Py4J-based Java parser server used by BRaIn to extract methods from Java source files.

## Overview

The BRaIn pipeline uses a Java parser to extract methods from source code. There are two options:

1. **Py4J Java Server (Recommended)**: Uses JavaParser library via Py4J for more accurate parsing
2. **Fallback Parser**: Uses `javalang` (pure Python) when Java server is unavailable

The script `a_Cache_initial_search_files.py` will automatically use the fallback if the Java server is not running.

## Setup Java Server (Recommended)

### Prerequisites

- Java JDK 8 or higher
- Py4J Python package (already in requirements.txt)
- Python 3.10

### Quick Setup

1. **Run the setup script:**
   ```bash
   cd src/java
   ./setup_java_server.sh
   ```

   This will:
   - Extract Java source files from `astParser.zip`
   - Compile the Java code
   - Create a startup script

2. **Start the Java server:**
   ```bash
   ./start_java_server.sh
   ```

   The server will start and listen on port 25333 (default Py4J port).

3. **Run in background (optional):**
   ```bash
   nohup ./start_java_server.sh > /tmp/java_server.log 2>&1 &
   ```

### Manual Setup

If the setup script doesn't work, you can set it up manually:

1. **Extract Java files:**
   ```bash
   unzip astParser.zip -d /tmp/astParser
   ```

2. **Find Py4J JAR:**
   ```bash
   python3 -c "import py4j; import os; print(os.path.join(os.path.dirname(py4j.__file__), 'java_gateway.jar'))"
   ```

3. **Compile Java code:**
   ```bash
   cd /tmp/astParser/me/asif/astparser
   javac -cp "<path_to_py4j_jar>" -d <build_dir> *.java
   ```

4. **Run the server:**
   ```bash
   cd <build_dir>
   java -cp ".:<path_to_py4j_jar>" me.asif.astparser.AstEntryPoint
   ```

## Using Without Java Server (Fallback)

If you don't want to set up the Java server, the script will automatically use the fallback parser (`JavaSourceParser` using `javalang`). This is a pure Python parser that doesn't require a Java server.

**Note**: The fallback parser may be less accurate than the JavaParser-based server, but it should work for most cases.

## Troubleshooting

### Connection Refused Error

If you see `ConnectionRefusedError` or `Py4JNetworkError`:

1. **Check if server is running:**
   ```bash
   lsof -i :25333
   ```

2. **Start the server** (see Quick Setup above)

3. **Or let the script use the fallback** - it will automatically fall back to `JavaSourceParser`

### Port Already in Use

If port 25333 is already in use:

1. Find the process using the port:
   ```bash
   lsof -i :25333
   ```

2. Kill the process or use a different port (requires modifying the Java code)

### Java Compilation Errors

If compilation fails:

1. Check Java version: `java -version` (should be 8+)
2. Ensure Py4J JAR is accessible
3. Check that all Java source files are present

## Files

- `astParser.zip`: Compressed Java source files
- `astParserJAVA.zip`: Alternative Java source archive
- `setup_java_server.sh`: Automated setup script
- `start_java_server.sh`: Server startup script (created by setup)

## Architecture

The Java server uses:
- **JavaParser**: A Java library for parsing Java source code
- **Py4J**: Bridge between Python and Java
- **GatewayServer**: Py4J server that listens on port 25333

The Python script connects to this server via Py4J's `JavaGateway` to parse Java source files and extract methods.
