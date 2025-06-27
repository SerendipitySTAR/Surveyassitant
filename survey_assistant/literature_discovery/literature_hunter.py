import requests
import feedparser
import time
import logging
import re # Added for query cleaning
from urllib.parse import quote_plus
from datetime import datetime
import xml.etree.ElementTree as ET # For parsing PubMed XML

# Assuming utils.py is in survey_assistant package, one level up
from ..utils import load_config # For cache directory from config
import os # For path operations for cache
from diskcache import Cache # For API call caching


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize cache for LiteratureHunter API calls
config = load_config()
cache_base_dir = config.get("cache", {}).get("base_dir", "cache")
lh_cache_dir = os.path.join(cache_base_dir, "literature_hunter_api")
if not os.path.exists(lh_cache_dir):
    try:
        os.makedirs(lh_cache_dir)
    except OSError as e:
        logger.error(f"Could not create cache directory {lh_cache_dir}: {e}")
        lh_cache_dir = os.path.join(cache_base_dir, "lh_temp_default")
        if not os.path.exists(lh_cache_dir): os.makedirs(lh_cache_dir, exist_ok=True)
# Cache API responses for 1 day (24 * 60 * 60 seconds)
api_cache = Cache(lh_cache_dir, expire=24 * 60 * 60, tag_index=True)


class LiteratureHunter:
    def __init__(self, user_agent="SurveyAssistant/0.1"):
        self.user_agent = user_agent
        # Config is already loaded at module level for cache setup
        # self.config = load_config()

        pubmed_config = config.get("literature_sources", {}).get("pubmed", {})
        self.pubmed_api_key = pubmed_config.get("api_key") # NCBI API key
        self.pubmed_email = pubmed_config.get("email_for_api", "surveyassistant@example.com") # Default email

        self.sources = {
            "arxiv": self._fetch_arxiv,
            "pubmed": self._fetch_pubmed,
        }
        logger.info(f"LiteratureHunter initialized with User-Agent: {self.user_agent}. PubMed API key configured: {'Yes' if self.pubmed_api_key else 'No'}. API Cache at: {lh_cache_dir}")

    def _parse_kpi_target(self, kpi_target: str) -> int:
        default_papers = 50
        num_papers_target = default_papers
        try:
            if "≥" in kpi_target:
                num_papers_target = int(kpi_target.split("≥")[1].split(" ")[0])
            elif ">" in kpi_target:
                num_papers_target = int(kpi_target.split(">")[1].split(" ")[0]) + 1
            elif "=" in kpi_target: # Allow exact numbers too
                num_papers_target = int(kpi_target.split("=")[1].split(" ")[0])
        except (ValueError, IndexError, AttributeError):
            logger.warning(f"Could not parse KPI target '{kpi_target}', defaulting to {default_papers} papers.")
        return num_papers_target

    def _fetch_arxiv(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Fetches papers from arXiv API.
        query: Search query (e.g., "electron OR ion")
        max_results: Maximum number of results to return.
        """
        base_url = "http://export.arxiv.org/api/query?"
        # Query format: ti:electron+AND+au:Smith
        # For general search, use 'all:' or just the terms. arXiv default is 'all:'.
        # Replace special characters and encode query for URL
        # Using 'all:' prefix for a general search query.
        # More complex queries can be built: https://arxiv.org/help/api/user-manual#query_details

        # Simple cleaning for iterative refinement patterns
        cleaned_query = re.sub(r'\s*\(iterative refinement\)\s*$', '', query, flags=re.IGNORECASE).strip()
        cleaned_query = re.sub(r'\s*\(refined search\)\s*$', '', cleaned_query, flags=re.IGNORECASE).strip()

        search_query = f"all:{quote_plus(cleaned_query)}"

        # Other params: sortBy (relevance, lastUpdatedDate, submittedDate), sortOrder (ascending, descending)
        params = {
            "search_query": search_query,
            "start": 0, # Note: Caching needs to consider pagination if start > 0
            "max_results": max_results,
            "sortBy": "relevance"
        }

        # Use a combination of query and key params for the cache key
        cache_key = f"arxiv_{cleaned_query}_{max_results}_{params['sortBy']}"

        cached_response_content = api_cache.get(cache_key)

        if cached_response_content:
            logger.info(f"Cache HIT for arXiv query: '{query}'. Parsing from cache.")
            response_content = cached_response_content
        else:
            logger.info(f"Cache MISS for arXiv query: '{query}'. Querying API.")
            try:
                response = requests.get(base_url, params=params, headers={"User-Agent": self.user_agent}, timeout=15) # Added timeout
                response.raise_for_status()
                response_content = response.content
                api_cache.set(cache_key, response_content) # Cache the raw byte content
                # arXiv API guidelines: wait 3 seconds between requests.
                time.sleep(1) # Being polite. More critical if making many unique calls.
            except requests.exceptions.RequestException as e:
                logger.error(f"arXiv API request failed for query '{query}': {e}")
                return []

        papers = []
        try:
            feed = feedparser.parse(response_content)
            if feed.bozo:
                 logger.warning(f"Feedparser encountered issues parsing arXiv response (key: {cache_key}). Bozo type: {feed.bozo_exception}")

            for entry in feed.entries:
                arxiv_id = entry.get("id", "").split('/')[-1] # e.g. http://arxiv.org/abs/2303.00001v1 -> 2303.00001v1
                title = entry.get("title", "N/A").strip().replace('\n', ' ').replace('\r', '')
                abstract = entry.get("summary", "N/A").strip().replace('\n', ' ').replace('\r', '')

                authors = [author.get("name") for author in entry.get("authors", [])]

                published_date_str = entry.get("published") # Format like '2023-03-15T18:00:00Z'
                updated_date_str = entry.get("updated")

                year = None
                if published_date_str:
                    try:
                        year = datetime.strptime(published_date_str, "%Y-%m-%dT%H:%M:%SZ").year
                    except ValueError:
                        logger.warning(f"Could not parse year from published_date: {published_date_str} for {arxiv_id}")

                categories = [tag.get("term") for tag in entry.get("tags", [])]

                # Get PDF link (usually entry.id is the abstract page, link with type 'application/pdf' is the PDF)
                pdf_link = None
                for link in entry.get("links", []):
                    if link.get("type") == "application/pdf":
                        pdf_link = link.get("href")
                        break
                if not pdf_link: # Fallback if PDF link not found directly (e.g., some older entries)
                    pdf_link = entry.get("id", "").replace("/abs/", "/pdf/") # Common pattern

                paper_data = {
                    "id": f"arxiv:{arxiv_id}", # Prefix with source
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": year,
                    "source": "arXiv",
                    "url": entry.get("link", entry.get("id")), # Link to abstract page
                    "pdf_url": pdf_link,
                    "published_date": published_date_str,
                    "updated_date": updated_date_str,
                    "categories": categories, # arXiv specific categories/tags
                    "keywords": [], # arXiv API doesn't directly provide keywords in the same way some other dbs do
                                    # Categories can serve a similar purpose.
                    # Fields for other agents
                    "github_url": None, # To be potentially found by other means or manual input
                    "experiment_setup": None
                }
                papers.append(paper_data)

            logger.info(f"Fetched {len(papers)} papers from arXiv for query '{query}'.")

            # arXiv API guidelines: wait 3 seconds between requests.
            # If making multiple calls in rapid succession, this is important.
            # For a single fetch_papers call, it's less critical unless it's part of a loop.
            time.sleep(1) # Being polite, even for single calls. Rate limits are per IP.

        except requests.exceptions.RequestException as e:
            logger.error(f"arXiv API request failed: {e}")
        except Exception as e:
            logger.error(f"Error processing arXiv results: {e}")

        return papers

    def _fetch_pubmed(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Fetches papers from PubMed using E-utilities.
        query: Search query.
        max_results: Maximum number of results to return.
        """
        base_esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        base_efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        papers = []

        # Simple cleaning for iterative refinement patterns
        cleaned_query = re.sub(r'\s*\(iterative refinement\)\s*$', '', query, flags=re.IGNORECASE).strip()
        cleaned_query = re.sub(r'\s*\(refined search\)\s*$', '', cleaned_query, flags=re.IGNORECASE).strip()

        esearch_params = {
            "db": "pubmed",
            "term": cleaned_query,
            "retmax": max_results,
            "usehistory": "y", # To use history for efetch
            "tool": "SurveyAssistant", # As per NCBI guidelines
            "email": self.pubmed_email # As per NCBI guidelines
        }
        if self.pubmed_api_key:
            esearch_params["api_key"] = self.pubmed_api_key

        esearch_cache_key = f"pubmed_esearch_{cleaned_query}_{max_results}"
        cached_esearch_content = api_cache.get(esearch_cache_key)
        response_esearch_content = None

        if cached_esearch_content:
            logger.info(f"Cache HIT for PubMed esearch: '{query}'. Parsing from cache.")
            response_esearch_content = cached_esearch_content
        else:
            logger.info(f"Cache MISS for PubMed esearch: '{query}'. Querying API.")
            try:
                response_esearch = requests.get(base_esearch_url, params=esearch_params, headers={"User-Agent": self.user_agent}, timeout=15)
                response_esearch.raise_for_status()
                response_esearch_content = response_esearch.content
                api_cache.set(esearch_cache_key, response_esearch_content)
                time.sleep(0.2 if self.pubmed_api_key else 0.4)
            except requests.exceptions.RequestException as e:
                logger.error(f"PubMed esearch API request failed for query '{query}': {e}")
                return []

        try:
            esearch_root = ET.fromstring(response_esearch_content)
            id_list = [id_elem.text for id_elem in esearch_root.findall(".//IdList/Id")]
            if not id_list:
                logger.info(f"No PubMed IDs found for query '{query}'.")
                return []

            web_env = esearch_root.find(".//WebEnv")
            query_key = esearch_root.find(".//QueryKey")

            if web_env is None or query_key is None:
                logger.error("Could not get WebEnv or QueryKey from PubMed esearch response.")
                # Could fall back to fetching IDs one by one, but less efficient.
                # For now, just return if history isn't available.
                # Alternative: join id_list and pass as comma-separated 'id' param to efetch if history fails.
                # This is less robust for large ID lists.
                id_string = ",".join(id_list)
                efetch_params_no_history = {
                    "db": "pubmed",
                    "id": id_string,
                    "retmode": "xml", # Abstract and metadata
                    "tool": "SurveyAssistant",
                    "email": self.pubmed_email
                }
                if self.pubmed_api_key:
                    efetch_params_no_history["api_key"] = self.pubmed_api_key

                logger.info(f"PubMed esearch did not return WebEnv/QueryKey. Fetching {len(id_list)} PMIDs directly.")
                efetch_api_params = efetch_params_no_history
            else: # Use history
                efetch_api_params = {
                    "db": "pubmed", "WebEnv": web_env.text, "query_key": query_key.text,
                    "retstart": 0, "retmax": len(id_list), "retmode": "xml",
                    "tool": "SurveyAssistant", "email": self.pubmed_email
                }
                if self.pubmed_api_key: efetch_api_params["api_key"] = self.pubmed_api_key
                logger.info(f"Querying PubMed (efetch) for {len(id_list)} PMIDs using history.")

            # Caching for efetch (based on sorted list of IDs to ensure consistent key)
            sorted_id_string = ",".join(sorted(id_list))
            efetch_cache_key = f"pubmed_efetch_{hash(sorted_id_string)}" # Hash long string of IDs

            cached_efetch_content = api_cache.get(efetch_cache_key)
            response_efetch_content = None

            if cached_efetch_content:
                logger.info(f"Cache HIT for PubMed efetch (IDs: {sorted_id_string[:50]}...). Parsing from cache.")
                response_efetch_content = cached_efetch_content
            else:
                logger.info(f"Cache MISS for PubMed efetch (IDs: {sorted_id_string[:50]}...). Querying API.")
                try:
                    response_efetch = requests.get(base_efetch_url, params=efetch_api_params, headers={"User-Agent": self.user_agent}, timeout=30) # Longer timeout for potentially many IDs
                    response_efetch.raise_for_status()
                    response_efetch_content = response_efetch.content
                    api_cache.set(efetch_cache_key, response_efetch_content)
                    time.sleep(0.2 if self.pubmed_api_key else 0.4)
                except requests.exceptions.RequestException as e:
                    logger.error(f"PubMed efetch API request failed for IDs '{sorted_id_string[:50]}...': {e}")
                    return [] # Return empty if efetch fails

            efetch_root = ET.fromstring(response_efetch_content)

            for article_elem in efetch_root.findall(".//PubmedArticle"):
                pmid_elem = article_elem.find(".//MedlineCitation/PMID")
                pmid = pmid_elem.text if pmid_elem is not None else None

                title_elem = article_elem.find(".//ArticleTitle")
                title = "".join(title_elem.itertext()).strip() if title_elem is not None else "N/A"

                abstract_text = []
                for abst_text_elem in article_elem.findall(".//Abstract/AbstractText"):
                    label = abst_text_elem.get("Label")
                    text_content = "".join(abst_text_elem.itertext()).strip()
                    if label:
                        abstract_text.append(f"{label}: {text_content}")
                    else:
                        abstract_text.append(text_content)
                abstract = " ".join(abstract_text) if abstract_text else "N/A"
                if not abstract or abstract == "N/A": # Try OtherAbstract if main Abstract is empty
                    other_abstract_elem = article_elem.find(".//OtherAbstract/AbstractText")
                    if other_abstract_elem is not None:
                        abstract = "".join(other_abstract_elem.itertext()).strip()


                authors_list = []
                for author_elem in article_elem.findall(".//AuthorList/Author"):
                    lastname = author_elem.find("LastName")
                    forename = author_elem.find("ForeName")
                    initials = author_elem.find("Initials") # Sometimes ForeName is missing

                    name_parts = []
                    if forename is not None and forename.text: name_parts.append(forename.text)
                    elif initials is not None and initials.text: name_parts.append(initials.text) # Use initials if forename missing
                    if lastname is not None and lastname.text: name_parts.append(lastname.text)
                    if name_parts: authors_list.append(" ".join(name_parts))

                year = None
                # Try to get year from PubDate
                pub_date_year_elem = article_elem.find(".//PubDate/Year")
                if pub_date_year_elem is not None and pub_date_year_elem.text:
                    year = int(pub_date_year_elem.text)
                else: # Fallback to MedlineDate if Year is not available
                    medline_date_elem = article_elem.find(".//PubDate/MedlineDate")
                    if medline_date_elem is not None and medline_date_elem.text:
                        # MedlineDate format can be like "2000 Spring" or "2000 Dec 24-30" or "2000"
                        match = requests.re.search(r'(\d{4})', medline_date_elem.text) # Extract first 4-digit number
                        if match: year = int(match.group(1))


                journal_title_elem = article_elem.find(".//Journal/Title")
                journal_title = journal_title_elem.text if journal_title_elem is not None else "N/A"

                mesh_headings = []
                for mesh_elem in article_elem.findall(".//MeshHeadingList/MeshHeading"):
                    descriptor_name_elem = mesh_elem.find(".//DescriptorName")
                    if descriptor_name_elem is not None and descriptor_name_elem.text:
                        mesh_headings.append(descriptor_name_elem.text)

                doi_elem = article_elem.find(".//ArticleIdList/ArticleId[@IdType='doi']")
                doi = doi_elem.text if doi_elem is not None else None

                paper_data = {
                    "id": f"pubmed:{pmid}",
                    "title": title,
                    "abstract": abstract,
                    "authors": authors_list,
                    "year": year,
                    "source": "PubMed",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                    "pdf_url": None, # PubMed doesn't directly provide PDF links in this API response easily.
                                     # Often requires linking out via DOI or PMC if available.
                    "journal": journal_title,
                    "mesh_terms": mesh_headings, # PubMed specific MeSH terms
                    "doi": doi,
                    "keywords": mesh_headings, # Use MeSH terms as keywords for consistency
                     # Fields for other agents
                    "github_url": None,
                    "experiment_setup": None
                }
                papers.append(paper_data)

            logger.info(f"Fetched and processed {len(papers)} articles from PubMed for query '{query}'.")

        except requests.exceptions.RequestException as e:
            logger.error(f"PubMed API request failed: {e}")
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed XML response: {e}")
        except Exception as e:
            logger.error(f"Error processing PubMed results: {e}")

        return papers


    def fetch_papers(self, query: str, kpi_target: str = "≥50 papers", sources: list[str] = None) -> list[dict]:
        """
        Fetches papers based on the query from specified sources.
        kpi_target: A string like "≥50 papers" to indicate desired number.
        sources: List of sources to query (e.g., ["arxiv", "pubmed"]). Defaults to all configured.
        """
        if sources is None:
            sources = list(self.sources.keys()) # Default to all available sources

        num_papers_target_total = self._parse_kpi_target(kpi_target)
        # Distribute target among sources, or fetch max from each and then select
        # For simplicity, let's aim for num_papers_target_total from each source if multiple,
        # then deduplicate and trim. A smarter way would be to divide the target.
        # For now, let's use a simpler approach: aim for roughly target/len(sources) from each.

        target_per_source = max(10, num_papers_target_total // len(sources) if sources else num_papers_target_total)


        all_papers = []
        for source_name in sources:
            if source_name in self.sources:
                fetch_func = self.sources[source_name]
                logger.info(f"Fetching from {source_name} for query '{query}' (target per source: {target_per_source})")
                try:
                    source_papers = fetch_func(query, max_results=target_per_source)
                    all_papers.extend(source_papers)
                    logger.info(f"Fetched {len(source_papers)} papers from {source_name}.")
                except Exception as e:
                    logger.error(f"Failed to fetch papers from {source_name}: {e}")
            else:
                logger.warning(f"Source '{source_name}' not recognized or implemented.")

        # Deduplication (simple version based on ID, or title+year if IDs are very different)
        # A more robust deduplication would normalize titles, compare DOIs if available, etc.
        unique_papers_dict = {}
        for paper in all_papers:
            # Prefer ID for uniqueness, but title+year can be a fallback
            # arXiv IDs are usually unique like "arxiv:xxxx.xxxxxvx"
            paper_key = paper.get("id")
            if not paper_key: # Fallback if ID is missing (should not happen with current arXiv parser)
                 paper_key = (paper.get("title", "").lower(), paper.get("year"))

            if paper_key not in unique_papers_dict:
                unique_papers_dict[paper_key] = paper

        unique_papers_list = list(unique_papers_dict.values())

        logger.info(f"Total unique papers fetched: {len(unique_papers_list)} across {len(sources)} source(s). Target was ~{num_papers_target_total}.")

        # If we have too many, we might want to sort by relevance (if available) and trim.
        # arXiv results are already sorted by relevance by default if not specified otherwise.
        # For now, just return up to the original total target if we exceed it significantly.
        # Or, perhaps return all unique ones found up to a reasonable cap.
        # Let's cap at num_papers_target_total * 1.5 for now if we got too many.
        if len(unique_papers_list) > num_papers_target_total * 1.5:
            logger.info(f"Trimming results from {len(unique_papers_list)} to {num_papers_target_total} (approx).")
            # This simple trim isn't ideal as it doesn't preserve cross-source relevance ranking.
            # A better approach would be to fetch more, then re-rank/select.
            # For now, a simple slice is fine for the current implementation stage.
            return unique_papers_list[:num_papers_target_total]

        return unique_papers_list

if __name__ == '__main__':
    hunter = LiteratureHunter()

    # Test arXiv fetching
    print("\n--- Testing arXiv Fetch ---")
    # query_arxiv = "transformer language model" # Broad query
    query_arxiv = "LLM agent memory" # More specific
    # query_arxiv = "cat:cs.AI AND LLM" # Query with category
    # results_arxiv = hunter.fetch_papers(query_arxiv, kpi_target="≥5 papers", sources=["arxiv"])
    results_arxiv = hunter._fetch_arxiv(query_arxiv, max_results=3) # Test specific method

    if results_arxiv:
        print(f"\nFetched {len(results_arxiv)} papers from arXiv for query '{query_arxiv}':")
        for i, res in enumerate(results_arxiv):
            print(f"  Paper {i+1}:")
            print(f"    ID: {res.get('id')}")
            print(f"    Title: {res.get('title')}")
            print(f"    Authors: {', '.join(res.get('authors', []))}")
            print(f"    Year: {res.get('year')}")
            # print(f"    Abstract: {res.get('abstract', '')[:100]}...") # Print first 100 chars
            print(f"    URL: {res.get('url')}")
            print(f"    PDF URL: {res.get('pdf_url')}")
            print(f"    Categories: {res.get('categories')}")
    else:
        print(f"No results from arXiv for query '{query_arxiv}'.")

    print("\n--- Testing Combined Fetch (currently only arXiv implemented) ---")
    query_combined = "artificial intelligence in healthcare"
    results_combined = hunter.fetch_papers(query_combined, kpi_target="=10 papers", sources=["arxiv"]) # Requesting 10

    if results_combined:
        print(f"\nFetched {len(results_combined)} papers in combined search for '{query_combined}':")
        for res in results_combined[:2]: # Print details of first 2
             print(f"  ID: {res.get('id')}, Title: {res.get('title')}, Year: {res.get('year')}")
    else:
        print(f"No results from combined search for '{query_combined}'.")

    # Example of a query that might return few or no results
    print("\n--- Testing Query with Few/No Results ---")
    query_specific_no_results = "nonexistent topic qwertyuiopasdfghjkl"
    results_no_results = hunter.fetch_papers(query_specific_no_results, kpi_target="≥5 papers", sources=["arxiv", "pubmed"])
    if not results_no_results:
        print(f"Correctly found no results for highly specific/nonexistent query: '{query_specific_no_results}'.")
    else:
        print(f"Found {len(results_no_results)} for query '{query_specific_no_results}', expected none.")

    print("\n--- Testing PubMed Fetch ---")
    query_pubmed = "crispr gene editing ethics"
    # results_pubmed = hunter.fetch_papers(query_pubmed, kpi_target="≥5 papers", sources=["pubmed"])
    results_pubmed = hunter._fetch_pubmed(query_pubmed, max_results=2) # Test specific method

    if results_pubmed:
        print(f"\nFetched {len(results_pubmed)} papers from PubMed for query '{query_pubmed}':")
        for i, res in enumerate(results_pubmed):
            print(f"  Paper {i+1}:")
            print(f"    ID: {res.get('id')}")
            print(f"    Title: {res.get('title')}")
            print(f"    Authors: {', '.join(res.get('authors', []))}")
            print(f"    Year: {res.get('year')}")
            print(f"    Journal: {res.get('journal')}")
            # print(f"    Abstract: {res.get('abstract', '')[:100]}...")
            print(f"    URL: {res.get('url')}")
            print(f"    DOI: {res.get('doi')}")
            print(f"    MeSH Terms: {res.get('mesh_terms')}")
    else:
        print(f"No results from PubMed for query '{query_pubmed}'.")


    print("\n--- Testing Combined Fetch (arXiv and PubMed) ---")
    query_combined_all = "diabetes management using wearable sensors"
    results_combined_all = hunter.fetch_papers(query_combined_all, kpi_target="=10 papers", sources=["arxiv", "pubmed"])

    if results_combined_all:
        print(f"\nFetched {len(results_combined_all)} papers in combined search for '{query_combined_all}':")
        for res in results_combined_all[:3]: # Print details of first 3
             print(f"  ID: {res.get('id')}, Source: {res.get('source')}, Title: {res.get('title')}, Year: {res.get('year')}")
    else:
        print(f"No results from combined search for '{query_combined_all}'.")
