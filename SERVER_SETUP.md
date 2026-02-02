# Server Setup Guide for BRaIn

## Step 1: Create the Elasticsearch Index

On the server, run:

```bash
cd ~/BRaIn
source venv/bin/activate
python -c "import sys; sys.path.insert(0, 'src'); from IR.Indexer.Index_Creator import Index_Creator; creator = Index_Creator(); creator.create_index(delete_if_exists=False)"
```

This creates the `defects4j` index with the correct schema.

## Step 2: Index Defects4J Source Code

You need to transfer and run the indexing script on the server:

### Transfer the indexing script:
```bash
# From your local machine
scp defects4j/index_defects4j_source.py user@server:~/BRaIn/defects4j/
scp defects4j/get_files.py user@server:~/BRaIn/defects4j/
```

### On the server, index the source code:
```bash
cd ~/BRaIn
source venv/bin/activate

# Make sure you have the Defects4J checkouts on the server
# If not, you'll need to checkout the buggy versions:
# cd defects4j
# ./checkout.sh

# Then index:
python defects4j/index_defects4j_source.py
```

This will index all the Java source files from your Defects4J checkouts into Elasticsearch.

## Step 3: Verify Index

Check that documents are indexed:

```bash
curl http://localhost:9200/defects4j/_count
```

You should see a count of indexed documents (should be ~211,000+ files).

## Step 4: Run the Caching Script

Now you can run:

```bash
python src/BRaIn/a_Cache_initial_search_files.py
```

---

## Alternative: Use Remote Elasticsearch

If you want to use the Elasticsearch from your local machine:

1. **Update config on server:**
   Edit `src/IR/config/IR_config.yaml`:
   ```yaml
   elasticsearch:
     host: YOUR_LOCAL_MACHINE_IP  # Change from localhost
     port: 9200
   ```

2. **Make sure Elasticsearch is accessible:**
   - Check firewall rules
   - Elasticsearch might need to bind to `0.0.0.0` instead of `127.0.0.1`
   - Update `elasticsearch.yml`:
     ```yaml
     network.host: 0.0.0.0
     ```
