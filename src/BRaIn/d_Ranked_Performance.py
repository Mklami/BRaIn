import sys
from pathlib import Path
from tqdm import tqdm

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
        
        # Track which bugs can/can't be localized
        if checkGTExists(ground_truths, search_results):
            localized_bugs.append({
                'bug_id': bug_id,
                'project': project,
                'sub_project': sub_project,
                'version': version,
                'bug_title': bug_title[:100] if bug_title else '',  # Truncate for readability
                'fixed_files': ground_truths,
                'top_10_results': search_results
            })
        else:
            non_localized_bugs.append({
                'bug_id': bug_id,
                'project': project,
                'sub_project': sub_project,
                'version': version,
                'bug_title': bug_title[:100] if bug_title else '',
                'fixed_files': ground_truths,
                'top_10_results': search_results
            })

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

    print(f"\nWhole Performance: {performance} Refined Total Bug Reports: {len(refined_gt)}")
    print(f"Found Gt for {count_of_found_gt} number of files")
    
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