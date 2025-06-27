import nltk
import re
import logging
import os # Added for path operations
# from sklearn.feature_extraction.text import TfidfVectorizer
# from ..knowledge_base.knowledge_base import KnowledgeBase
from diskcache import Cache
from ..utils import load_config # For cache directory from config

logger = logging.getLogger(__name__)

# Initialize cache for DeepAnalyzer results
# Cache directory can be configured via config.yaml
config = load_config()
cache_base_dir = config.get("cache", {}).get("base_dir", "cache")
analysis_cache_dir = config.get("cache", {}).get("analysis_cache_dir", os.path.join(cache_base_dir, "analysis"))
# Ensure directory exists (utils.ensure_dir_exists could be used if module load order allows)
if not os.path.exists(analysis_cache_dir):
    try:
        os.makedirs(analysis_cache_dir)
    except OSError as e:
        logger.error(f"Could not create cache directory {analysis_cache_dir}: {e}")
        # Fallback to a temporary cache or disable caching if dir creation fails
        analysis_cache_dir = os.path.join(cache_base_dir, "analysis_temp_default") # Default temp if error
        if not os.path.exists(analysis_cache_dir): os.makedirs(analysis_cache_dir, exist_ok=True)

analysis_cache = Cache(analysis_cache_dir)


# Ensure NLTK sentence tokenizer is available
# Moved to a global setup step in the main workflow or agent initialization.
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     logger.info("NLTK 'punkt' tokenizer not found. Downloading...")
#     nltk.download('punkt', quiet=True)

class DeepAnalyzer:
    def __init__(self, paper_data: dict, kb=None): # kb: KnowledgeBase = None
        """
        Initializes the DeepAnalyzer with a specific paper's data.
        paper_data: A dictionary containing information about the paper (e.g., from LiteratureHunter).
        kb: Optional KnowledgeBase for contextual information or existing analyses.
        """
        self.paper_data = paper_data
        self.kb = kb # Not used in this basic version, but available for future extension
        logger.info(f"DeepAnalyzer initialized for paper: {paper_data.get('id', 'Unknown ID')}.")

    def _nltk_sentence_tokenize(self, text: str) -> list[str]:
        """Tokenizes text into sentences using NLTK."""
        try:
            return nltk.sent_tokenize(text)
        except Exception as e:
            logger.error(f"NLTK sentence tokenization failed: {e}. Falling back to simple split.")
            return text.split('. ') # Basic fallback

    def _extract_by_keywords(self, sentences: list[str], patterns: dict) -> dict:
        """
        Extracts sentences matching a list of keyword patterns.
        patterns: A dictionary where keys are categories (e.g., "contribution")
                  and values are lists of regex patterns.
        """
        extracted_info = {key: [] for key in patterns}
        for sentence in sentences:
            for category, regex_list in patterns.items():
                for regex_pattern in regex_list:
                    if re.search(regex_pattern, sentence, re.IGNORECASE):
                        extracted_info[category].append(sentence.strip())
                        break # Avoid matching same sentence to multiple patterns in the same category
        return extracted_info

    def _generate_structured_summary(self, sentences: list[str], max_sentences=3) -> str:
        """Generates a very basic summary from the first few sentences of the abstract."""
        # This is a naive summary. LLMs would do much better.
        summary_sentences = sentences[:max_sentences]
        return " ".join(summary_sentences)

    def _identify_potential_claims(self, sentences: list[str], paper_id: str) -> list[dict]:
        """
        Identifies potential claims from sentences using heuristic patterns.
        More advanced claim detection would use ML models.
        """
        claims = []
        # Keywords that might indicate a claim or finding
        claim_keywords = [
            r"demonstrates that", r"shows that", r"indicates that", r"results suggest",
            r"we found that", r"our study reveals", r"it is clear that",
            r"proves that", r"confirms that", r"validates that",
            r"is superior to", r"outperforms", r"more effective than", r"significant improvement",
            r"leads to", r"results in", r"causes a", r"effect of"
        ]
        # Negative keywords to avoid simple "we aim to show that..."
        non_claim_starts = [r"we aim to", r"this paper attempts to", r"the goal is to"]

        for i, sentence in enumerate(sentences):
            is_potential_claim = False
            source_snippet = sentence[:150] + "..." if len(sentence) > 150 else sentence # Keep snippet short

            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in claim_keywords):
                if not any(re.match(nc_pattern, sentence, re.IGNORECASE) for nc_pattern in non_claim_starts):
                    is_potential_claim = True

            if is_potential_claim:
                claims.append({
                    "claim_id": f"{paper_id}_claim_{len(claims) + 1}",
                    "text": sentence.strip(),
                    "confidence": 0.6,  # Default confidence for rule-based extraction
                    "source_text_snippet": source_snippet
                })
        return claims

    def _extract_methodologies(self, text_content: str, paper_keywords: list) -> list[str]:
        """
        Extracts methodologies using keywords from the paper and a predefined list.
        This is a simplistic approach. ML/NER would be better.
        """
        # Predefined general ML/CS methodology terms (very basic list)
        common_methods = [
            "machine learning", "deep learning", "neural network", "cnn", "rnn", "lstm", "transformer",
            "support vector machine", "svm", "random forest", "decision tree", "bayesian network",
            "statistical analysis", "regression analysis", "simulation", "algorithm", "framework",
            "natural language processing", "nlp", "computer vision", "reinforcement learning",
            "experimental study", "case study", "survey", "literature review", "meta-analysis",
            "qualitative analysis", "quantitative analysis", "control group", "monte carlo"
        ]

        found_methods = set()
        # Add paper's own keywords (e.g., from arXiv categories or MeSH terms) to the search list
        # These are often good indicators of methodologies or primary topics.
        searchable_terms = common_methods + [kw.lower() for kw in paper_keywords if isinstance(kw, str)]

        text_lower = text_content.lower()
        for method in searchable_terms:
            if re.search(r"\b" + re.escape(method) + r"\b", text_lower): # Whole word match
                found_methods.add(method.capitalize()) # Capitalize for display

        # If paper keywords were used and are specific, they might be better than generic terms.
        # This needs refinement to prioritize. For now, just collect all.
        return sorted(list(found_methods))


    def _extract_llm_placeholder(self, text_section: str, prompt_type: str) -> list[str] | str:
        """Placeholder for where an LLM would be called for specific extraction tasks."""
        logger.info(f"Placeholder: LLM call for '{prompt_type}' on text: '{text_section[:100]}...'")
        if prompt_type == "summarize_abstract":
            return f"[LLM Summary Placeholder: {text_section[:50]}...]"
        elif prompt_type == "extract_contributions":
            return ["[LLM Contribution Placeholder 1]", "[LLM Contribution Placeholder 2]"]
        # Add more types as needed
        # In a real scenario, this would involve formatting a prompt, sending to an LLM API or local model,
        # and parsing the response.
        # For simulation, we generate plausible-looking but fixed/randomized output.

        if prompt_type == "summarize_abstract":
            # Simulate a concise summary
            summary_sentences = text_section.split('.')[:2] # Take first two sentences as a base
            return f"[Simulated LLM Summary]: {' '.join(s.strip() for s in summary_sentences if s.strip())}."

        elif prompt_type == "extract_contributions":
            # Simulate extracting a few bullet points
            return [
                "[Simulated LLM] Contribution: Developed a novel XYZ technique.",
                "[Simulated LLM] Contribution: Achieved SOTA results on Benchmark A.",
                f"[Simulated LLM] Contribution: Reduced computational cost by X% for '{text_section[:20]}...'"
            ]

        elif prompt_type == "extract_limitations":
            return [
                "[Simulated LLM] Limitation: Study was conducted on a limited dataset.",
                f"[Simulated LLM] Limitation: Generalizability to other domains not explored for '{text_section[:20]}...'"
            ]

        elif prompt_type == "extract_future_work":
            return [
                "[Simulated LLM] Future Work: Explore scalability of the proposed method.",
                f"[Simulated LLM] Future Work: Apply technique to real-world problem Y related to '{text_section[:20]}...'"
            ]

        logger.warning(f"Unknown prompt_type '{prompt_type}' for LLM simulation.")
        return "LLM Simulation: No specific output for this prompt type."

    # The main run method will call this internal, memoized method.
    # The cache key will be based on paper_id.
    # Note: If paper_data content (other than ID) could change for the same ID,
    # this caching strategy would be problematic. Assuming paper_id uniquely identifies content.
    @analysis_cache.memoize(tag="deep_analysis_v1") # Added tag for versioning cache
    def _perform_analysis_memoized(self, paper_id: str, title: str, abstract: str, keywords: list, categories: list, mesh_terms: list) -> dict:
        """
        Internal method that performs the actual analysis. This method is memoized.
        All data relevant for analysis that comes from self.paper_data must be passed as arguments
        to ensure the cache key is comprehensive.
        """
        logger.info(f"DeepAnalyzer (Cache Miss): Performing analysis for '{title}' (ID: {paper_id}).")

        if not abstract: # Should be caught by caller, but double check.
            logger.warning(f"No abstract provided for paper {paper_id} to _perform_analysis_memoized.")
            return {
                "paper_id": paper_id, "title": title, "structured_summary": "N/A - No abstract provided.",
                "key_contributions": [], "limitations": [], "methodologies_used": [],
                "datasets_used": [], "experimental_results_summary": "N/A",
                "potential_claims": [], "future_work_suggestions": []
            }

        sentences = self._nltk_sentence_tokenize(abstract)

        # LLM-simulated extractions (preferred if available, fallback to regex/keywords)
        # For now, we'll assume LLM simulation provides the primary content for these.
        structured_summary = self._extract_llm_placeholder(abstract, "summarize_abstract")
        key_contributions_llm = self._extract_llm_placeholder(abstract, "extract_contributions")
        limitations_llm = self._extract_llm_placeholder(abstract, "extract_limitations")
        future_work_llm = self._extract_llm_placeholder(abstract, "extract_future_work")

        # Keyword-based extraction for things LLM might not focus on or as fallback
        extraction_patterns = {
            # Contributions/limitations might be superseded by LLM, but keep for robustness/detail
            # "key_contributions_regex": [
            #     r"our main contribution", r"we propose", r"novel method", r"key finding",
            #     r"we introduce", r"this paper presents", r"significantly improves"
            # ],
            # "limitations_regex": [
            #     r"limitation of this study", r"however,.*limited to", r"future work should address",
            #     r"drawback is", r"one challenge is", r"does not consider"
            # ],
            # "future_work_suggestions_regex": [
            #     r"future work", r"further research", r"potential extension", r"next steps involve"
            # ],
            "experimental_results_summary_indicators": [
                r"results show", r"experiments demonstrate", r"achieved an accuracy of", r"outperformed",
                r"validate our approach", r"evaluation reveals"
            ],
            "datasets_used_indicators": [
                r"dataset used", r"evaluated on", r"tested on the", r"benchmark dataset[s]?",
                r"using the .* dataset"
            ]
        }
        extracted_keyword_sections = self._extract_by_keywords(sentences, extraction_patterns)

        experimental_results_summary = " ".join(extracted_keyword_sections.get("experimental_results_summary_indicators", []))

        datasets_identified = []
        for hint_sentence in extracted_keyword_sections.get("datasets_used_indicators", []):
            matches = re.findall(r"([A-Z][A-Za-z0-9]*(?:[\s-][A-Z][A-Za-z0-9]*)* dataset)", hint_sentence)
            if matches:
                datasets_identified.extend([m.replace(" dataset", "") for m in matches])
            elif len(hint_sentence) < 100: # Keep it brief if no specific pattern
                 datasets_identified.append(f"Hint: {hint_sentence}")

        paper_keywords = self.paper_data.get('keywords', []) + self.paper_data.get('categories', []) + self.paper_data.get('mesh_terms', [])
        methodologies = self._extract_methodologies(abstract, paper_keywords)

        potential_claims = self._identify_potential_claims(sentences, paper_id)

        analysis_result = {
            "paper_id": paper_id,
            "title": title,
            "structured_summary": structured_summary, # Primarily from LLM sim
            "key_contributions": key_contributions_llm if isinstance(key_contributions_llm, list) else [key_contributions_llm],
            "limitations": limitations_llm if isinstance(limitations_llm, list) else [limitations_llm],
            "methodologies_used": methodologies, # Still regex/keyword based
            "datasets_used": list(set(datasets_identified)) if datasets_identified else [],
            "experimental_results_summary": experimental_results_summary if experimental_results_summary else "Not explicitly extracted by keyword patterns.",
            "potential_claims": potential_claims, # Still regex/keyword based
            "future_work_suggestions": future_work_llm if isinstance(future_work_llm, list) else [future_work_llm]
        }

        logger.info(f"DeepAnalyzer: Analysis complete for '{title}'. Found {len(potential_claims)} potential claims (rule-based). LLM parts simulated.")
        return analysis_result

    def run(self) -> dict:
        """
        Public method to run analysis. It retrieves paper data from self.paper_data
        and calls the memoized internal method.
        """
        paper_id = self.paper_data.get('id', 'Unknown ID')
        title = self.paper_data.get('title', 'N/A')
        abstract = self.paper_data.get('abstract', '')

        # For robust caching, ensure all relevant parts of paper_data used by
        # _perform_analysis_memoized are passed to it.
        keywords = self.paper_data.get('keywords', [])
        categories = self.paper_data.get('categories', [])
        mesh_terms = self.paper_data.get('mesh_terms', [])

        logger.info(f"DeepAnalyzer: Requesting analysis for '{title}' (ID: {paper_id}). Checking cache...")

        # Handle case where abstract is missing before calling memoized function
        if not abstract:
            logger.warning(f"No abstract found for paper {paper_id} in run(). Analysis will be limited.")
            return {
                "paper_id": paper_id, "title": title, "structured_summary": "N/A - No abstract provided.",
                "key_contributions": [], "limitations": [], "methodologies_used": [],
                "datasets_used": [], "experimental_results_summary": "N/A",
                "potential_claims": [], "future_work_suggestions": []
            }

        # Call the memoized method
        # The cache key will be formed from these arguments.
        result = self._perform_analysis_memoized(paper_id, title, abstract, keywords, categories, mesh_terms)

        # Check if the result came from cache by inspecting logs or by a more direct way if diskcache offered it.
        # For now, the logger message inside _perform_analysis_memoized indicates a cache miss.
        # A simple way to check for hit (outside of diskcache's own logging which isn't easily captured here):
        # If the "Cache Miss" log for this paper_id for this run doesn't appear a second time, it was a hit.
        return result


if __name__ == '__main__':
    # Setup basic logging for standalone run
    logging.basicConfig(level=logging.INFO)
    logger.info("NLTK 'punkt' and 'stopwords' should be pre-downloaded for DeepAnalyzer.")
    # Example: nltk.download('punkt'); nltk.download('stopwords')

    sample_paper_1_data = {
        "id": "arxiv:sample001_v2", # Changed ID to test caching
        "title": "A Novel Method for Topic Modeling using Deep Learning",
        "abstract": ("This paper presents a novel method for topic modeling. Our main contribution is a new neural architecture "
                     "that significantly improves coherence scores. We evaluated on the 20 Newsgroups dataset. "
                     "Results show our approach outperforms existing state-of-the-art methods by 15%. "
                     "A limitation of this study is its computational cost. Future work should explore efficiency."),
        "keywords": ["topic modeling", "deep learning", "neural networks"],
        "categories": ["cs.AI", "cs.LG"],
        "mesh_terms": []
    }
    analyzer1 = DeepAnalyzer(paper_data=sample_paper_1_data)

    print("\n--- Deep Analysis Result 1 (Run 1) ---")
    result1_run1 = analyzer1.run()
    import json
    print(json.dumps(result1_run1, indent=2))

    print("\n--- Deep Analysis Result 1 (Run 2 - Should be cached) ---")
    # If caching works, the "_perform_analysis_memoized" log for "Cache Miss" should not appear here.
    result1_run2 = analyzer1.run()
    print(json.dumps(result1_run2, indent=2))
    assert result1_run1 == result1_run2 # Results should be identical if from cache or re-computed

    sample_paper_2_data = {
        "id": "pubmed:sample002_v2",
        "title": "The Effect of Drug X on Protein Y",
        "abstract": ("We investigated the effect of Drug X on Protein Y in vitro. Our study reveals that Drug X inhibits Protein Y activity. "
                     "Experiments demonstrate a dose-dependent response. This is a key finding for cancer research. "
                     "However, the study was limited to cell cultures. Further research is needed in vivo. "
                     "The data was analyzed using statistical analysis and regression models. "
                     "This work validates the potential of Drug X."),
        "keywords": [], "categories": [],
        "mesh_terms": ["Drug X", "Protein Y", "Neoplasms", "Cell Line"]
    }
    analyzer2 = DeepAnalyzer(paper_data=sample_paper_2_data)
    print("\n--- Deep Analysis Result 2 ---")
    result2 = analyzer2.run()
    print(json.dumps(result2, indent=2))

    sample_paper_no_abstract_data = {
        "id": "special:noabs003_v2",
        "title": "A Paper With No Abstract",
        "abstract": "",
        "keywords": [], "categories": [], "mesh_terms": []
    }
    analyzer3 = DeepAnalyzer(paper_data=sample_paper_no_abstract_data)
    print("\n--- Deep Analysis Result 3 (No Abstract) ---")
    result3 = analyzer3.run()
    print(json.dumps(result3, indent=2))

    logger.info("DeepAnalyzer tests finished. Check logs for 'Cache Miss' messages to verify caching behavior.")
