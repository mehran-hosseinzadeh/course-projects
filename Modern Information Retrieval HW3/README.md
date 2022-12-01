# Modern Information Retrieval HW3

In this project, I have implemented a scraper (crawler) on papers on Semantic Scholar. Starting from an intial seed, the scraper extracts indormation about each paper (id, title, authors, date, references, and abstract). These papers are stroed in the file ***papers.json***.

Functions named ***add_data_to_elasticsearch*** and ***delete_data_from_elasticsearch*** are designed as interfaces for data storing and indexing in ElasticSearch.

PageRank algorithm is further employed on these data in ***calculate_page_rank*** function.

A function named ***search*** is also incorporated for searching on ElasticSearch documents and retrieving documents given a query. 

The top authors (based on references and links between papers) are also ranked using HITS (Hyperlink_Induced Topic Search) algorithm in ***rate_authors*** function.

For using all of these tools, an interactive command line is also desgined that may be used by running the python file.


