# Defects4J Migration Guide

This guide outlines what data you need from the Defects4J dataset to adapt BRaIn from Bench4BL (B4BL) to Defects4J.

## Required Data from Defects4J

### 1. Bug Report Data (JSON Format)

You need to create a JSON file similar to `Data/Refined_B4BL.json` with the following structure:

```json
[
  {
    "bug_id": "Chart-1",
    "bug_title": "Bug title from Defects4J",
    "bug_description": "Bug description from Defects4J",
    "project": "Chart",
    "sub_project": "Chart",  // See mapping section below
    "version": "1b",  // Buggy version identifier
    "fixed_version": "1f",  // Fixed version identifier (optional)
    "fixed_files": [
      "org/jfree/chart/renderer/category/AbstractCategoryItemRenderer.java",
      "org/jfree/chart/renderer/category/BarRenderer.java"
    ]
  }
]
```

#### Required Fields:

1. **`bug_id`** (string): 
   - Defects4J format: `{Project}-{BugID}` (e.g., "Chart-1", "Lang-1")
   - Used for: Bug identification and tracking

2. **`bug_title`** (string):
   - Source: Defects4J bug report summary/title
   - Used for: Query construction (`bug_title + ' . ' + bug_description`)

3. **`bug_description`** (string):
   - Source: Defects4J bug report description
   - Used for: Query construction and IR search

4. **`project`** (string):
   - Defects4J format: Project name (e.g., "Chart", "Lang", "Math", "Time", "Closure")
   - Used for: Elasticsearch filtering and indexing

5. **`sub_project`** (string):
   - **Note**: Defects4J doesn't have sub_projects like B4BL
   - **Solution**: Map `sub_project` = `project` (use same value as `project`)
   - Used for: Elasticsearch filtering (required by current code structure)

6. **`version`** (string):
   - Defects4J format: Buggy version identifier (e.g., "1b", "2b")
   - Used for: Elasticsearch filtering to search correct version
   - Alternative: Could use full version string if available

7. **`fixed_version`** (string, optional):
   - Defects4J format: Fixed version identifier (e.g., "1f", "2f")
   - Used for: Some processing scripts (may not be critical)

8. **`fixed_files`** (array of strings):
   - **Critical for evaluation**: List of file paths that were actually fixed
   - Defects4J format: Relative paths from project root (e.g., `"org/jfree/chart/renderer/category/BarRenderer.java"`)
   - Used for: Ground truth in evaluation (MAP, MRR, HIT@K metrics)
   - **Important**: File paths must match the format used in Elasticsearch `file_url` field

### 2. Source Code Data for Indexing

You need to extract and index source code files for each buggy version of each project.

#### For Each Project-Bug Combination:

1. **Extract Source Code**:
   - Checkout the buggy version: `defects4j checkout -p {Project} -v {BugID}b -w {working_dir}`
   - Extract all Java source files from the project

2. **Index in Elasticsearch** with these fields:
   - `project`: Project name (e.g., "Chart")
   - `sub_project`: Same as project (e.g., "Chart")
   - `version`: Buggy version identifier (e.g., "1b")
   - `source_code`: Full content of the Java source file
   - `file_url`: File path (must match format in `fixed_files`)

#### File Path Format Consistency:

**Critical**: The `file_url` in Elasticsearch must match the format used in `fixed_files` array. 

- Defects4J typically uses paths like: `org/jfree/chart/renderer/category/BarRenderer.java`
- Ensure consistency between:
  - How you store `file_url` when indexing
  - How you store paths in `fixed_files` array
  - How the evaluation script compares them

## Data Extraction from Defects4J

### Step 1: Extract Bug Reports

Use Defects4J API to extract bug information:

```bash
# For each project and bug
defects4j info -p Chart -b 1
```

Extract:
- Bug summary (title)
- Bug description
- Fixed files list

### Step 2: Create JSON Dataset

Write a script to convert Defects4J data to the required JSON format:

```python
# Pseudo-code structure
bugs = []
for project in defects4j_projects:
    for bug_id in range(1, max_bugs+1):
        bug_info = defects4j.get_bug_info(project, bug_id)
        fixed_files = defects4j.get_fixed_files(project, bug_id)
        
        bug_entry = {
            "bug_id": f"{project}-{bug_id}",
            "bug_title": bug_info.summary,
            "bug_description": bug_info.description,
            "project": project,
            "sub_project": project,  # Map to same as project
            "version": f"{bug_id}b",  # Buggy version
            "fixed_version": f"{bug_id}f",  # Fixed version
            "fixed_files": fixed_files  # List of file paths
        }
        bugs.append(bug_entry)
```

### Step 3: Extract and Index Source Code

For each buggy version:

```bash
# Checkout buggy version
defects4j checkout -p Chart -v 1b -w /tmp/chart_1b

# Extract all Java files and index them
# Use Indexer.py to index each file with:
# - project: "Chart"
# - sub_project: "Chart"
# - version: "1b"
# - source_code: file content
# - file_url: relative path from project root
```

## Key Differences: B4BL vs Defects4J

| Aspect | B4BL | Defects4J |
|--------|------|-----------|
| Project Structure | `project` + `sub_project` | Only `project` |
| Version Format | Version strings (e.g., "camel-1.3.0") | Buggy/Fixed versions (e.g., "1b", "1f") |
| Bug ID Format | Numeric or custom | `{Project}-{BugID}` |
| File Paths | Various formats | Relative from project root |
| Bug Reports | From GitHub issues | From bug tracking system |

## Code Modifications Needed

### Minimal Changes Required:

1. **Mapping `sub_project`**: 
   - Since Defects4J doesn't have sub_projects, set `sub_project = project` in your data extraction script

2. **File Path Normalization**:
   - Ensure consistent path format between `fixed_files` and `file_url` in Elasticsearch
   - Consider normalizing paths (e.g., always use forward slashes, relative paths)

3. **Version Handling**:
   - Current code expects `version` field - use buggy version identifier (e.g., "1b")
   - Or modify code to handle Defects4J version format if needed

### Optional Enhancements:

1. **Update Config Files**:
   - Update `IR_config.yaml` if index name needs to change
   - Consider renaming index from "bench4bl" to "defects4j"

2. **Path Matching Logic**:
   - Review `d_Ranked_Performance.py` to ensure file path matching works correctly
   - The `checkGTExists()` function compares `fixed_files` with search result `file_url`s

## Testing Your Migration

1. **Verify JSON Structure**:
   - Load your Defects4J JSON file using `JSON_File_IO.load_JSON_to_Dataframe()`
   - Ensure all required fields are present

2. **Verify Indexing**:
   - Check that source code is indexed correctly
   - Verify `file_url` format matches `fixed_files` format

3. **Test Search**:
   - Run `a_Cache_initial_search_files.py` with a small subset
   - Verify search results return correct files

4. **Test Evaluation**:
   - Run `d_Ranked_Performance.py` 
   - Verify ground truth matching works correctly

## Summary Checklist

- [ ] Extract bug reports from Defects4J (title, description, fixed files)
- [ ] Create JSON file with required fields (bug_id, bug_title, bug_description, project, sub_project, version, fixed_files)
- [ ] Map `sub_project` = `project` for Defects4J
- [ ] Extract source code for each buggy version
- [ ] Index source code in Elasticsearch with correct fields
- [ ] Ensure `file_url` format matches `fixed_files` format
- [ ] Test with small subset before full migration
- [ ] Update configuration files if needed

## Additional Resources

- Defects4J Documentation: https://github.com/rjust/defects4j
- Defects4J API: `defects4j info -p {Project} -b {BugID}`
- Defects4J Checkout: `defects4j checkout -p {Project} -v {BugID}b -w {dir}`
