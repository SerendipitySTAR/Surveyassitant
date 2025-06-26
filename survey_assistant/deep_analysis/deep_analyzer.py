import nltk
import re
import logging
# from sklearn.feature_extraction.text import TfidfVectorizer # For more advanced keyword extraction
# from ..knowledge_base.knowledge_base import KnowledgeBase # May need KB for context

logger = logging.getLogger(__name__)

# Ensure NLTK sentence tokenizer is available (download if first time)
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
        return "LLM Placeholder: No specific output."


    def run(self) -> dict:
        """
        Performs deep analysis on the paper's abstract using rule-based methods.
        """
        paper_id = self.paper_data.get('id', 'Unknown ID')
        title = self.paper_data.get('title', 'N/A')
        abstract = self.paper_data.get('abstract', '')

        logger.info(f"DeepAnalyzer: Running analysis on '{title}' (ID: {paper_id}).")

        if not abstract:
            logger.warning(f"No abstract found for paper {paper_id}. Analysis will be limited.")
            # Return a minimal structure or the placeholder if no abstract
            return {
                "paper_id": paper_id, "title": title, "structured_summary": "N/A - No abstract provided.",
                "key_contributions": [], "limitations": [], "methodologies_used": [],
                "datasets_used": [], "experimental_results_summary": "N/A",
                "potential_claims": [], "future_work_suggestions": []
            }

        sentences = self._nltk_sentence_tokenize(abstract)

        # Define patterns for keyword-based extraction
        # These regex patterns are examples and would need refinement.
        extraction_patterns = {
            "key_contributions": [
                r"our main contribution", r"we propose", r"novel method", r"key finding",
                r"we introduce", r"this paper presents", r"significantly improves"
            ],
            "limitations": [
                r"limitation of this study", r"however,.*limited to", r"future work should address",
                r"drawback is", r"one challenge is", r"does not consider"
            ],
            "future_work_suggestions": [
                r"future work", r"further research", r"potential extension", r"next steps involve"
            ],
            "experimental_results_summary_indicators": [ # Sentences that might talk about results
                r"results show", r"experiments demonstrate", r"achieved an accuracy of", r"outperformed",
                r"validate our approach", r"evaluation reveals"
            ],
            "datasets_used_indicators": [
                r"dataset used", r"evaluated on", r"tested on the", r"benchmark dataset[s]?",
                r"using the .* dataset"
            ]
        }

        extracted_sections = self._extract_by_keywords(sentences, extraction_patterns)

        # Consolidate results (simple concatenation for now)
        # For results/datasets, these are just indicative sentences, not structured data.
        experimental_results_summary = " ".join(extracted_sections.get("experimental_results_summary_indicators", []))
        datasets_used_hints = extracted_sections.get("datasets_used_indicators", [])

        # More sophisticated dataset extraction would parse specific names.
        # For placeholder, just list sentences that hint at datasets.
        datasets_identified = []
        for hint_sentence in datasets_used_hints:
            # Try to find capitalized words or phrases that might be dataset names
            # This is very naive. NER or more specific patterns would be better.
            matches = re.findall(r"([A-Z][A-Za-z0-9]*(?:[\s-][A-Z][A-Za-z0-9]*)* dataset)", hint_sentence)
            if matches:
                datasets_identified.extend([m.replace(" dataset", "") for m in matches])
            else: # If no specific "X Dataset" pattern, add the sentence as a general hint
                if len(hint_sentence) < 100: # Keep it brief
                     datasets_identified.append(f"Hint: {hint_sentence}")


        # Methodologies: Use paper keywords (like arXiv categories or MeSH terms) + common list
        paper_keywords = self.paper_data.get('keywords', []) + self.paper_data.get('categories', []) + self.paper_data.get('mesh_terms', [])
        methodologies = self._extract_methodologies(abstract, paper_keywords)

        # Potential Claims
        potential_claims = self._identify_potential_claims(sentences, paper_id)

        # Structured Summary (very basic for now)
        # Replace with LLM call in future: self._extract_llm_placeholder(abstract, "summarize_abstract")
        structured_summary = self._generate_structured_summary(sentences)


        analysis_result = {
            "paper_id": paper_id,
            "title": title,
            "structured_summary": structured_summary,
            "key_contributions": extracted_sections.get("key_contributions", []),
            "limitations": extracted_sections.get("limitations", []),
            "methodologies_used": methodologies,
            "datasets_used": list(set(datasets_identified)) if datasets_identified else [], # Unique dataset mentions
            "experimental_results_summary": experimental_results_summary if experimental_results_summary else "Not explicitly extracted.",
            "potential_claims": potential_claims,
            "future_work_suggestions": extracted_sections.get("future_work_suggestions", [])
        }

        logger.info(f"DeepAnalyzer: Analysis complete for '{title}'. Found {len(potential_claims)} potential claims.")
        return analysis_result

if __name__ == '__main__':
    # Setup basic logging for standalone run
    logging.basicConfig(level=logging.INFO)

    sample_paper_1 = {
        "id": "arxiv:sample001",
        "title": "A Novel Method for Topic Modeling using Deep Learning",
        "abstract": ("This paper presents a novel method for topic modeling. Our main contribution is a new neural architecture "
                     "that significantly improves coherence scores. We evaluated on the 20 Newsgroups dataset. "
                     "Results show our approach outperforms existing state-of-the-art methods by 15%. "
                     "A limitation of this study is its computational cost. Future work should explore efficiency."),
        "keywords": ["topic modeling", "deep learning", "neural networks"],
        "categories": ["cs.AI", "cs.LG"]
    }
    analyzer1 = DeepAnalyzer(paper_data=sample_paper_1)
    result1 = analyzer1.run()

    print("\n--- Deep Analysis Result 1 ---")
    import json
    print(json.dumps(result1, indent=2))

    sample_paper_2 = {
        "id": "pubmed:sample002",
        "title": "The Effect of Drug X on Protein Y",
        "abstract": ("We investigated the effect of Drug X on Protein Y in vitro. Our study reveals that Drug X inhibits Protein Y activity. "
                     "Experiments demonstrate a dose-dependent response. This is a key finding for cancer research. "
                     "However, the study was limited to cell cultures. Further research is needed in vivo. "
                     "The data was analyzed using statistical analysis and regression models. "
                     "This work validates the potential of Drug X."),
        "mesh_terms": ["Drug X", "Protein Y", "Neoplasms", "Cell Line"]
    }
    analyzer2 = DeepAnalyzer(paper_data=sample_paper_2)
    result2 = analyzer2.run()

    print("\n--- Deep Analysis Result 2 ---")
    print(json.dumps(result2, indent=2))

    sample_paper_no_abstract = {
        "id": "special:noabs003",
        "title": "A Paper With No Abstract",
        "abstract": "" # Empty abstract
    }
    analyzer3 = DeepAnalyzer(paper_data=sample_paper_no_abstract)
    result3 = analyzer3.run()
    print("\n--- Deep Analysis Result 3 (No Abstract) ---")
    print(json.dumps(result3, indent=2))
