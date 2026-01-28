import json
import sys
import os
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = script_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from tqdm import tqdm

from Utils import JavaSourceParser
from Utils.IO import JSON_File_IO
from IR import Searcher
from Utils.Parser import SourceRefiner


def load_dataframe(file_path):
    return JSON_File_IO.load_JSON_to_Dataframe(file_path)


def load_json_to_dict(file_path):
    return JSON_File_IO.load_JSON_to_Dict(file_path)


def perform_search(project, sub_project, version, bug_title, bug_description, top_K_results=10):
    # Use index name from config (defaults to 'defects4j' for Defects4J)
    searcher = Searcher()  # Will use index from config file
    search_results = searcher.search_Extended(
        project=project,
        sub_project=sub_project,
        version=version,
        query=bug_title + '. ' + bug_description,
        top_K_results=top_K_results,
        field_to_return=["file_url", "source_code"]
    )

    # print(search_results)
    return search_results


# Initialize Py4J gateway (optional - will use fallback if unavailable)
java_py4j_ast_parser = None
try:
    from py4j.java_gateway import JavaGateway
    gateway = JavaGateway()  # connect to the JVM
    java_py4j_ast_parser = gateway.entry_point.getJavaMethodParser()
    print("Py4J Java parser connected successfully")
except Exception as e:
    print(f"Warning: Py4J Java server not available ({e}). Using fallback JavaSourceParser.")
    java_py4j_ast_parser = None


def search_result_ops(search_results):
    processed_results = []
    for result in search_results:
        file_url = result['file_url']
        source_code = result['source_code']
        bm25_score = result['bm25_score']

        # Try Py4J parser first, fallback to JavaSourceParser if unavailable
        json_result = None
        if java_py4j_ast_parser is not None:
            try:
                json_result = java_py4j_ast_parser.processJavaFileContent(source_code)
            except Exception as e:
                # Py4J connection failed, will use fallback
                json_result = None

        if json_result is None or json_result == '':
            # parse the source code if py4j fails or is unavailable
            javaParser = JavaSourceParser(data=source_code)
            parsed_methods = javaParser.parse_methods()

        else:
            loaded_json = json.loads(json_result)
            parsed_methods = {}

            poly_morphism = 1
            # iterate over the parsed methods and get the method names and the method bodies
            for method in loaded_json:

                method_name = method['member_name']
                method_body = method['member_body']
                class_name = method['class_name']

                # clear the formatting of the method body for tokenization
                method_body = SourceRefiner.clear_formatting(method_body)

                # check if the method name is already in the parsed_methods
                if method_name in parsed_methods:
                    # append the method body to the existing method name
                    parsed_methods[method_name+'!P'+str(poly_morphism)] = 'Class: '+ class_name + ' \n Method: ' + method_body
                    poly_morphism += 1
                else:
                    parsed_methods[method_name] = 'Class: '+ class_name + ' \n Method: ' + method_body



        # create a json object with file_url and parsed_methods
        json_object = {
            'file_url': file_url,
            'methods': parsed_methods,
            'bm25_score': bm25_score
        }

        processed_results.append(json_object)

    return processed_results


import html

if __name__ == '__main__':
    # Use absolute path relative to script location
    sample_path = script_dir / "Data" / "Refined_Defects4J.json"
    sample_path = str(sample_path)
    # load the json to dictionary
    df = load_dataframe(sample_path)



    # convert this to json string
    json_string = JSON_File_IO.convert_Dataframe_to_JSON_string(df)

    # iterate over the json string
    json_bugs = json.loads(json_string)

    chunk_size = 2350
    json_bugs_chunked = []

    # chunk the json_bugs up to chunk size or up to last if less than 1000
    for i in range(0, len(json_bugs), chunk_size):
        json_bugs_chunked.append(json_bugs[i:i + chunk_size])

    chunk_id = 1
    # iterate over the json_bugs_chunked
    for json_bugs in tqdm(json_bugs_chunked, desc="Processing JSON Bugs Chunked"):
        # iterate over the json array
        for bug in tqdm(json_bugs, desc="Processing JSON Bugs"):
        # for bug in json_bugs:
            bug_title = bug['bug_title']
            bug_description = bug['bug_description']
            project = bug['project']
            sub_project = bug['sub_project']
            version = bug['version']

            # now search for the query in a method
            search_results = perform_search(project, sub_project, version, bug_title, bug_description, top_K_results=50)

            # now, perform ops in the search results
            processed_results = search_result_ops(search_results)

            # add processed results to the json as a new key
            bug['es_results'] = processed_results

        # save the json to a file
        # Use relative path from project root
        json_save_path = str(script_dir / "Output" / "Cache" / "Chunked_50")
        #use chunk_id to save the file
        JSON_File_IO.save_Dict_to_JSON(json_bugs, json_save_path, "Cache_Res50_C"+str(chunk_id)+".json")
        chunk_id += 1

        # empty the json_bugs from memory after saving to save memory
        json_bugs = []

