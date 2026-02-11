import sys
import csv
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add src directory to Python path
script_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = script_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from Utils import Performance_Evaluator
from Utils.IO import JSON_File_IO

def checkGTExists(fixed_files, results):
    for file in fixed_files:
        if file in results:
            return True
    return False

def findFirstRank(fixed_files, search_results):
    """Find the first rank (1-indexed) at which any fixed file appears in search results.
    Returns None if no fixed file is found."""
    for rank, file_url in enumerate(search_results, start=1):
        if file_url in fixed_files:
            return rank
    return None

def calculateTopK(fixed_files, search_results, k):
    """Check if any fixed file appears in top-k results. Returns 1 if yes, 0 if no."""
    top_k_results = search_results[:k]
    return 1 if checkGTExists(fixed_files, top_k_results) else 0

count_of_found_gt = 0
if __name__ == '__main__':

    ##### train_test #####
    # Use absolute path relative to script location
    json_path = str(script_dir / "Output" / "Cache" / "Mistral_ZERO_sorted_cache.json")
    
    # Check if file exists, provide helpful error if not
    if not Path(json_path).exists():
        raise FileNotFoundError(
            f"Input JSON not found at: {json_path}\n"
            f"Expected it under Output/Cache/. Make sure you've run step 2c (c_PRF_Scoring_cache.py) first."
        )

    # load the json to dictionary
    json_bugs = JSON_File_IO.load_JSON_to_Dict(json_path)

    all_ground_truths = []
    all_search_results = []
    localized_bugs = []  # Bugs that can be localized (GT found in top-10)
    non_localized_bugs = []  # Bugs that cannot be localized (GT not found in top-10)

    for bug in tqdm(json_bugs, desc="Processing JSON Bugs"):
        bug_id = bug['bug_id']
        bug_title = bug['bug_title']
        bug_description = bug['bug_description']
        project = bug['project']
        sub_project = bug['sub_project']
        version = bug['version']
        responses = None
        # responses = bug['response']

        ground_truths = bug['fixed_files']
        search_results = []

        es_results = bug['es_results']

        for result in es_results[:10]:
            file_url = result['file_url']
            search_results.append(file_url)

        all_ground_truths.append(ground_truths)
        all_search_results.append(search_results)
        
        # Find the rank at which this bug was localized
        first_rank = findFirstRank(ground_truths, search_results)
        is_localized = first_rank is not None
        
        bug_data = {
            'bug_id': bug_id,
            'project': project,
            'sub_project': sub_project,
            'version': version,
            'bug_title': bug_title[:100] if bug_title else '',
            'fixed_files': ground_truths,
            'top_10_results': search_results,
            'rank': first_rank,  # None if not localized, otherwise 1-indexed rank
            'top@1': calculateTopK(ground_truths, search_results, 1),
            'top@5': calculateTopK(ground_truths, search_results, 5),
            'top@10': calculateTopK(ground_truths, search_results, 10)
        }
        
        # Track which bugs can/can't be localized
        if is_localized:
            localized_bugs.append(bug_data)
        else:
            non_localized_bugs.append(bug_data)

    gt_tracker_by_count = {}
    sr_tracker_by_count = {}
    for gt, sr in zip(all_ground_truths, all_search_results):
        # extract the file name from full . separated in gt and check gt contains string Test or test. if it does remove it.


        if checkGTExists(gt, sr):
            count_of_found_gt += 1


        if len(gt) <= 3:
            if len(gt) in gt_tracker_by_count:
                gt_tracker_by_count[len(gt)].append(gt)
                sr_tracker_by_count[len(gt)].append(sr)
            else:
                gt_tracker_by_count[len(gt)] = [gt]
                sr_tracker_by_count[len(gt)] = [sr]
        else:
            if 4 in gt_tracker_by_count:
                gt_tracker_by_count[4].append(gt)
                sr_tracker_by_count[4].append(sr)
            else:
                gt_tracker_by_count[4] = [gt]
                sr_tracker_by_count[4] = [sr]
    # evaluate the search results

    refined_gt = []
    refined_sr = []
    for key, value in gt_tracker_by_count.items():
        performance_evaluator = Performance_Evaluator()
        search_results = sr_tracker_by_count[key]
        performance = performance_evaluator.evaluate_several(value, search_results, at_Ks=[1, 5, 10])

        refined_gt.extend(value)
        refined_sr.extend(search_results)

        print(f"GT Count: {key} GT files: {len(value)} Performance: {performance}")

    # evaluate the search results
    performance_evaluator = Performance_Evaluator()
    performance = performance_evaluator.evaluate_several(refined_gt, refined_sr, at_Ks=[1, 5, 10])

    
    # Save localized and non-localized bugs to files
    output_dir = script_dir / "Output" / "Performance_Analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    localized_path = str(output_dir / "localized_bugs.json")
    non_localized_path = str(output_dir / "non_localized_bugs.json")
    
    JSON_File_IO.save_Dict_to_JSON(localized_bugs, str(output_dir), "localized_bugs.json", with_indent=True)
    JSON_File_IO.save_Dict_to_JSON(non_localized_bugs, str(output_dir), "non_localized_bugs.json", with_indent=True)
    
    print(f"\n{'='*80}")
    print(f"LOCALIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total bugs analyzed: {len(json_bugs)}")
    print(f"Successfully localized (GT in top-10): {len(localized_bugs)} ({len(localized_bugs)/len(json_bugs)*100:.1f}%)")
    print(f"Not localized (GT not in top-10): {len(non_localized_bugs)} ({len(non_localized_bugs)/len(json_bugs)*100:.1f}%)")
    print(f"\nResults saved to:")
    print(f"  - Localized bugs: {localized_path}")
    print(f"  - Non-localized bugs: {non_localized_path}")
    
    # Print sample of non-localized bugs
    if non_localized_bugs:
        print(f"\n{'='*80}")
        print(f"SAMPLE OF NON-LOCALIZED BUGS (first 10):")
        print(f"{'='*80}")
        for i, bug in enumerate(non_localized_bugs[:10], 1):
            print(f"\n{i}. Bug ID: {bug['bug_id']}")
            print(f"   Project: {bug['project']} | Version: {bug['version']}")
            print(f"   Title: {bug['bug_title']}")
            print(f"   Fixed files: {bug['fixed_files']}")
            print(f"   Top result: {bug['top_10_results'][0] if bug['top_10_results'] else 'N/A'}")
    
    print(f"\n{'='*80}")
    print(f"OVERALL PERFORMANCE (All Bugs)")
    print(f"{'='*80}")
    print(f"Total bug reports analyzed: {len(refined_gt)}")
    print(f"Bugs with GT found in top-10: {count_of_found_gt} ({count_of_found_gt/len(refined_gt)*100:.1f}%)")
    print(f"\nMetrics:")
    print(f"  MAP:     {performance.get('map', 0):.4f}")
    print(f"  MRR:     {performance.get('mrr', 0):.4f}")
    print(f"  HIT@1:   {performance.get('hit@1', 0):.4f}")
    print(f"  HIT@5:   {performance.get('hit@5', 0):.4f}")
    print(f"  HIT@10:  {performance.get('hit@10', 0):.4f}")
    print(f"{'='*80}")
    
    # Export to CSV for tool comparison
    csv_path = script_dir / "tool_comparison_summary.csv"
    tool_name = "BRaIn"
    timestamp = datetime.now().isoformat()
    
    # Calculate MRR and MAP per bug for CSV
    csv_rows = []
    for bug in json_bugs:
        bug_id = bug['bug_id']
        project = bug['project']
        ground_truths = bug['fixed_files']
        search_results = [r['file_url'] for r in bug['es_results'][:10]]
        
        # Find rank
        first_rank = findFirstRank(ground_truths, search_results)
        detected = "Yes" if first_rank is not None else "No"
        rank_value = float(first_rank) if first_rank else None
        
        # Calculate MRR for this bug (1/rank if found, 0 otherwise)
        mrr_value = 1.0 / first_rank if first_rank else 0.0
        
        # Calculate MAP for this bug (average precision)
        # MAP = average of precisions at each relevant document position
        if first_rank:
            # For single GT file, MAP = 1/rank
            map_value = 1.0 / first_rank
        else:
            map_value = 0.0
        
        # Calculate top@K
        top1 = calculateTopK(ground_truths, search_results, 1)
        top5 = calculateTopK(ground_truths, search_results, 5)
        top10 = calculateTopK(ground_truths, search_results, 10)
        
        csv_rows.append({
            'project': project,
            'bug_id': bug_id,
            'tool': tool_name,
            'detected': detected,
            'rank': rank_value if rank_value else '',
            'mrr': mrr_value,
            'map': map_value,
            'duration_seconds': 'N/A',
            'execution_timestamp': timestamp,
            'comparison_timestamp': timestamp,
            'cpu_avg_percent': '',
            'cpu_max_percent': '',
            'memory_avg_bytes': '',
            'memory_max_bytes': '',
            'memory_limit_bytes': '',
            'network_rx_total_bytes': '',
            'network_tx_total_bytes': '',
            'disk_read_total_bytes': '',
            'disk_write_total_bytes': '',
            'top@1': top1,
            'top@5': top5,
            'top@10': top10
        })
    
    # Read existing CSV if it exists
    existing_rows = []
    if csv_path.exists():
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
    
    # Remove existing BRaIn entries for these bugs
    bug_keys = {(row['project'], row['bug_id']) for row in csv_rows}
    existing_rows = [row for row in existing_rows 
                     if not (row.get('project'), row.get('bug_id')) in bug_keys or row.get('tool') != tool_name]
    
    # Combine and write
    all_rows = existing_rows + csv_rows
    fieldnames = ['project', 'bug_id', 'tool', 'detected', 'rank', 'mrr', 'map', 'duration_seconds',
                  'execution_timestamp', 'comparison_timestamp', 'cpu_avg_percent', 'cpu_max_percent',
                  'memory_avg_bytes', 'memory_max_bytes', 'memory_limit_bytes', 'network_rx_total_bytes',
                  'network_tx_total_bytes', 'disk_read_total_bytes', 'disk_write_total_bytes',
                  'top@1', 'top@5', 'top@10']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\n{'='*80}")
    print(f"CSV EXPORT")
    print(f"{'='*80}")
    print(f"Added {len(csv_rows)} BRaIn entries to: {csv_path}")
    print(f"Total rows in CSV: {len(all_rows)}")
    print(f"{'='*80}")