import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import CloseSpider
import json
from elasticsearch import Elasticsearch
import numpy
import math


class Paper:
    def __init__(self, id, title, authors, date, abstract, references):
        self.id = id
        self.title = title
        self.authors = authors
        self.date = date
        self.references = references
        self.abstract = abstract

    def dump(self):
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "date": self.date,
            "abstract": self.abstract,
            "references": self.references,
        }


class SemanticScholarScraper(scrapy.Spider):
    number_of_crawled_pages = 2000
    name = "SemanticScholar_scraper"
    domain_url = "https://www.semanticscholar.org"
    start_urls = [
        "https://www.semanticscholar.org/paper/The-Lottery-Ticket-Hypothesis%3A-Training-Pruned-Frankle-Carbin/f90720ed12e045ac84beb94c27271d6fb8ad48cf",
        "https://www.semanticscholar.org/paper/Attention-is-All-you-Need-Vaswani-Shazeer/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        "https://www.semanticscholar.org/paper/BERT%3A-Pre-training-of-Deep-Bidirectional-for-Devlin-Chang/df2b0e26d0599ce3e70df8a9da02e51594e0e992"
    ]
    papers = []
    crawled_papers = []

    def parse(self, response):
        paper_id = response.url.split("/")[-1]
        if paper_id in self.crawled_papers:
            return

        try:
            paper_title = response.xpath("//meta[@name='citation_title']/@content")[0].get()
        except:
            paper_title = response.xpath("//meta[@property='citation_title']/@content")[0].get()

        try:
            paper_authors = [x.get() for x in response.xpath("//meta[@name='citation_author']/@content")]
        except:
            paper_authors = [x.get() for x in response.xpath("//meta[@property='citation_author']/@content")]

        try:
            paper_date = response.xpath("//meta[@name='citation_publication_date']/@content")[0].get()
        except:
            try:
                paper_date = response.xpath("//meta[@property='citation_publication_date']/@content")[0].get()
            except:
                paper_date = ""
        try:
            paper_abstract = response.xpath("//meta[@name='description']/@content")[0].get()
        except:
            paper_abstract = ""
        try:
            all_references = response.css(".card.references .citation__title a::attr(href)").extract()
            links = [self.domain_url + reference for reference in all_references]
            links = links[: min(10, len(links))]
            links = set(links)
            paper_references = [link.split("/")[-1] for link in links]
        except:
            links = []
            paper_references = []

        if len(self.papers) < self.number_of_crawled_pages:
            self.papers.append(
                Paper(paper_id, paper_title, paper_authors, paper_date, paper_abstract, paper_references))
            self.crawled_papers.append(paper_id)
            print(len(self.papers))
            for link in links:
                yield scrapy.Request(
                    response.urljoin(link),
                    callback=self.parse
                )
        else:

            with open("papers.json", 'w') as fp:
                json.dump([o.dump() for o in self.papers], fp)
            raise CloseSpider("Target number of papers Achieved!")


def add_data_to_elasticsearch(papers, elasticsearch_address):
    es = Elasticsearch(hosts=[
        {
            "host": elasticsearch_address.split(":")[0],
            "port": elasticsearch_address.split(":")[1]
        }
    ])
    for paper in papers:
        es.index(index="paper_index", id=paper["id"], body={"paper": paper})


def delete_data_from_elasticsearch(papers, elasticsearch_address):
    es = Elasticsearch(hosts=[
        {
            "host": elasticsearch_address.split(":")[0],
            "port": elasticsearch_address.split(":")[1]
        }
    ])
    es.delete_by_query(index="paper_index", body={"query": {"match_all": {}}})


def calculate_page_rank(elasticsearch_address, alpha):
    es = Elasticsearch(hosts=[
        {
            "host": elasticsearch_address.split(":")[0],
            "port": elasticsearch_address.split(":")[1]
        }
    ])
    id_to_index_dict = {}
    all_references = []
    number_of_papers = int(es.cat.count("paper_index", params={"format": "json"})[0]["count"])
    papers_data = es.search(
        index="paper_index", body={"query": {"match_all": {}}},
        size=number_of_papers
    )['hits']['hits']
    i = 0
    papers = []
    for paper_data in papers_data:
        paper = paper_data["_source"]["paper"]
        id_to_index_dict[paper["id"]] = i
        papers.append(paper)
        i += 1
    for paper in papers:
        paper_references = paper["references"]
        paper_references = list(filter(lambda x: x in id_to_index_dict, paper_references))
        if len(paper_references) == 0:
            all_references.append([(1.0 / number_of_papers) for _ in range(number_of_papers)])
        else:
            all_references.append([0 for _ in range(number_of_papers)])
            for reference in paper_references:
                index = id_to_index_dict[reference]
                all_references[-1][index] = \
                    (1 - alpha) * (1.0 / len(paper_references))
            for i in range(number_of_papers):
                all_references[-1][i] = all_references[-1][i] + (alpha * (1.0 / number_of_papers))
    p = numpy.array(all_references)
    a = [(1.0 / number_of_papers) for _ in range(number_of_papers)]
    a = numpy.array(a)
    last_norm = numpy.linalg.norm(a)
    i = 0
    while True:
        a = numpy.matmul(a, p)
        new_norm = numpy.linalg.norm(a)
        if math.fabs(new_norm - last_norm) < (10 ** (-20)):
            print("converged")
            break
        last_norm = new_norm
        i += 1
        print(a)
    for paper_id in id_to_index_dict:
        index = id_to_index_dict[paper_id]
        new_paper_info = papers[index]
        new_paper_info["page_rank"] = a[index]
        es.index(index="paper_index", id=paper_id, body={"paper": new_paper_info})


def search(elasticsearch_address, weights, texts, use_page_rank=False):
    es = Elasticsearch(hosts=[
        {
            "host": elasticsearch_address.split(":")[0],
            "port": elasticsearch_address.split(":")[1]
        }
    ])
    title_text = texts['title'] if 'title' in texts else ''
    abstract_text = texts['abstract'] if 'abstract' in texts else ''
    date_text = texts['date'] if 'date' in texts else ''
    title_weight = weights['title'] if 'title' in weights else 1
    abstract_weight = weights['abstract'] if 'abstract' in weights else 1
    date_weight = weights['date'] if 'date' in weights else 1
    search_query = {
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"paper.title": title_text}},
                            {"match": {"paper.abstract": abstract_text}},
                            {"range": {"paper.date": {"gte": date_text}}}
                        ]
                    }
                },
                "functions": [
                    {
                        "filter": {"match": {"paper.title": title_text}},
                        "weight": title_weight
                    },
                    {
                        "filter": {"match": {"paper.abstract": abstract_text}},
                        "weight": abstract_weight
                    },
                    {
                        "filter": {"range": {"paper.date": {"gte": date_text}}},
                        "weight": date_weight
                    }
                ],
                "score_mode": "sum",
            }
        }
    }
    papers_data = es.search(
        index="paper_index", body=search_query,
        size=10
    )['hits']['hits']

    papers = []
    for paper_data in papers_data:
        paper = paper_data["_source"]["paper"]
        papers.append(paper)
    return papers


def rate_authors(elasticsearch_address, n):
    es = Elasticsearch(hosts=[
        {
            "host": elasticsearch_address.split(":")[0],
            "port": elasticsearch_address.split(":")[1]
        }
    ])
    referrers = {}
    number_of_papers = int(es.cat.count("paper_index", params={"format": "json"})[0]["count"])
    papers_data = es.search(
        index="paper_index", body={"query": {"match_all": {}}},
        size=number_of_papers
    )['hits']['hits']
    author_auth = {}
    author_hub = {}
    papers = {}
    authors = {}
    for paper_data in papers_data:
        paper = paper_data["_source"]["paper"]
        papers[paper["id"]] = paper
        for author in paper['authors']:
            if author not in authors:
                authors[author] = {"references": []}

    for id in papers:
        paper = papers[id]
        for author in paper['authors']:
            for reference in paper["references"]:
                if reference in papers:
                    for referenced in papers[reference]["authors"]:
                        if referenced not in authors[author]["references"]:
                            authors[author]["references"].append(referenced)
    for author in authors:
        author_auth[author] = 1
        author_hub[author] = 1
        for reference in authors[author]["references"]:
            if reference not in referrers:
                referrers[reference] = [author]
            else:
                referrers[reference].append(author)
    for _ in range(5):
        norm = 0
        for author in authors:
            author_auth[author] = 0
            if author in referrers:
                for referrer in referrers[author]:
                    author_auth[author] += author_hub[referrer]
            norm += author_auth[author] ** 2
        norm = math.sqrt(norm)
        for author in author_auth:
            author_auth[author] = author_auth[author] / norm
        norm = 0
        for author in authors:
            author_hub[author] = 0
            for reference in authors[author]["references"]:
                author_hub[author] += author_auth[reference]
            norm += author_hub[author] ** 2
        norm = math.sqrt(norm)
        for author in author_hub:
            author_hub[author] = author_hub[author] / norm

    best_authors = [k for k, v in reversed(sorted(author_auth.items(), key=lambda item: item[1]))][:n]
    return best_authors

def crawl():
    print("If you wish to feed in the 3 initial links, enter y (Otherwise the default ones will be used)")
    c = input()
    start_urls = []
    if c == 'y':
        print("Please enter the links line by line")
        start_urls.append(input())
        start_urls.append(input())
        start_urls.append(input())
        SemanticScholarScraper.start_urls = start_urls
    print("Enter number of papers you wish to have: (Enter - to set 2000)")
    n = input()
    if n != "-":
        SemanticScholarScraper.number_of_crawled_pages = int(n)

    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })

    process.crawl(SemanticScholarScraper)
    process.start()
    print("Data Saved to data.json")


if __name__ == "__main__":

    print("Welcome to MIR Phase3 CLI")
    while True:
        print("Choose your option (Enter line number):")
        print("1. Run Crawler")
        print("2. Work With Elastic (data.json should be saved here in advance)")
        print("3. Calculate PageRank")
        print("4. Search Data")
        print("5. Show Top Authors")
        address = "localhost:9200"

        command = input()
        if command == '1':
            crawl()
        elif command == '2':
            print("Choose your action with elastic (Enter line number):")
            print("1. Add data to paper_index in elastic")
            print("2. Delete data from paper_index in elastic")
            c = input()
            if c == "1":
                with open("papers.json", 'r') as fp:
                    papers = json.load(fp)
                print("If you wish to give elastic address, enter y (Otherwise the localhost:9200 will be used)")
                c2 = input()
                if c2 == 'y':
                    print("Please enter the address:")
                    address = input()
                add_data_to_elasticsearch(papers, address)
                print("data saved to paper_index")
            elif c == "2":
                print("If you wish to give elastic address, enter y (Otherwise the localhost:9200 will be used)")
                c2 = input()
                if c2 == 'y':
                    print("Please enter the address:")
                    address = input()
                delete_data_from_elasticsearch({}, address)
                print("data removed from paper_index")
            else:
                print("Invalid Command")
        elif command == '3':
            address = "localhost:9200"
            print("If you wish to give elastic address, enter y (Otherwise the localhost:9200 will be used)")
            c2 = input()
            if c2 == 'y':
                print("Please enter the address:")
                address = input()
            alpha = 0.1
            print("If you wish to give alpha (between 0 and 1), enter y (Otherwise 0.1 will be used)")
            c2 = input()
            if c2 == 'y':
                print("Please enter the address:")
                alpha = int(input())
            calculate_page_rank(address, alpha)
        elif command == '4':
            print("If you wish to give elastic address, enter y (Otherwise the localhost:9200 will be used)")
            c2 = input()
            if c2 == 'y':
                print("Please enter the address:")
                address = input()

            weights = {}
            texts = {}
            print("Type in the text for title field: (Enter - to fill blank)")
            s = input()
            if s == '-':
                s = ''
            texts["title"] = s
            print("Type in the text for abstract field: (Enter - to fill blank)")
            s = input()
            if s == '-':
                s = ''
            texts["abstract"] = s
            print("Type in the text for date field: (Enter - to fill blank)")
            s = input()
            if s == '-':
                s = ''
            texts["date"] = s
            print("Type in the weight for title field: (Enter - to set 1)")
            s = input()
            if s == '-':
                s = 1
            s = int(s)
            weights["title"] = s
            print("Type in the weight for abstract field: (Enter - to set 1)")
            s = input()
            if s == '-':
                s = 1
            s = int(s)
            weights["abstract"] = s
            print("Type in the weight for date field: (Enter - to set 1)")
            s = input()
            if s == '-':
                s = 1
            s = int(s)
            weights["date"] = s
            results = search(address, weights, texts)
            print("Top 10 results are: (Only showing Title and Year. Results with full info is available in code)")
            i = 1
            for result in results:
                print(i, result["title"], result["date"])
                i = i + 1
        elif command == '5':
            print("If you wish to give elastic address, enter y (Otherwise the localhost:9200 will be used)")
            c2 = input()
            if c2 == 'y':
                print("Please enter the address:")
                address = input()
            print("Enter number of top authors you wish to have:")
            n = int(input())
            best_authors = rate_authors(address, n)
            print("Best Authors are:")
            i = 1
            for author in best_authors:
                print(i, author)
                i = i + 1
        print("******************************************************")
