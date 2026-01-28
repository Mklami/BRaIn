from elasticsearch import Elasticsearch, helpers

from IR.config.Elasic_Config_Loader import Elasic_Config_Loader


class Indexer:
    def __init__(self, index_name=None):
        """
        Initialize the indexer. Make sure the elastic search is running.
        :param index_name: No need to pass this parameter unless some particular reason. It will be loaded from config file.
        """

        # Create an instance of ConfigLoader (config file will be loaded automatically)
        self.bulk_index_array = []
        config_loader = Elasic_Config_Loader()

        # Accessing configuration parameters using class methods
        elastic_search_host = config_loader.get_elastic_search_host()
        elastic_search_port = config_loader.get_elastic_search_port()
        self.elastic_search_index = config_loader.get_index_name()

        if index_name is None:
            self.index_name = self.elastic_search_index
        else:
            self.index_name = index_name

        # Create an instance of Elasticsearch client
        self.es_client = Elasticsearch('http://' + elastic_search_host + ':' + str(elastic_search_port),
                                  # http_auth=("username", "password"),
                                  verify_certs=False)

    def index(self, project, sub_project, version, source_code, file_url):
        document = {
            "project": project,
            "sub_project": sub_project,
            "version": version,
            "source_code": source_code,
            "file_url": file_url
        }
        result = self.es_client.index(index=self.index_name, body=document, refresh=False)
        # print(f"Indexed document with ID: {result['_id']}")

        return result


    def bulk_action(self):
        for document in self.bulk_index_array:
            yield document

    # function for bulk indexing. it is same as before. saves the doc in array and when it reaches the limit, it indexes them in bulk
    def bulk_index(self, project, sub_project, version, source_code, file_url, bulk_size=1024):
        document = {
            "project": project,
            "sub_project": sub_project,
            "version": version,
            "source_code": source_code,
            "file_url": file_url
        }

        indexable_document = {
            "_index": self.index_name,
            "_source": document
        }
        self.bulk_index_array.append(indexable_document)

        if len(self.bulk_index_array) >= bulk_size:
            try:
                result = helpers.bulk(self.es_client, actions=self.bulk_action(), raise_on_error=False, stats_only=False)
                # helpers.bulk returns (success_count, failed_items) tuple
                if isinstance(result, tuple) and len(result) == 2:
                    success_count, failed_items = result
                    if failed_items:
                        print(f"Warning: {len(failed_items)} documents failed to index in bulk operation")
                        # Print first few errors for debugging
                        for error in failed_items[:3]:
                            error_info = error.get('index', {}).get('error', {})
                            error_msg = error_info.get('reason', 'Unknown error') if isinstance(error_info, dict) else str(error_info)
                            print(f"  Error: {error_msg}")
                    else:
                        print(f"Indexed {success_count} documents in bulk.")
                else:
                    # Fallback if return format is different
                    print(f"Indexed batch of {bulk_size} documents.")
            except Exception as e:
                print(f"Error in bulk indexing: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.bulk_index_array = []


    def refresh(self):
        if len(self.bulk_index_array) > 0:
            try:
                result = helpers.bulk(self.es_client, actions=self.bulk_action(), raise_on_error=False, stats_only=False)
                if isinstance(result, tuple) and len(result) == 2:
                    success_count, failed_items = result
                    if failed_items:
                        print(f"Warning: {len(failed_items)} documents failed to index in final flush")
                        for error in failed_items[:3]:
                            error_info = error.get('index', {}).get('error', {})
                            error_msg = error_info.get('reason', 'Unknown error') if isinstance(error_info, dict) else str(error_info)
                            print(f"  Error: {error_msg}")
                    else:
                        print(f"Indexed {success_count} documents in final flush.")
                else:
                    print(f"Indexed {len(self.bulk_index_array)} documents in final flush.")
            except Exception as e:
                print(f"Error in final bulk indexing: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.bulk_index_array = []

        self.es_client.indices.refresh(index=self.index_name)


    def __del__(self):
        self.refresh()
