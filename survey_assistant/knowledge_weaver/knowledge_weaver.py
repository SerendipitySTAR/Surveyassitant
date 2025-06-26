# from ..knowledge_base.knowledge_base import KnowledgeBase # Import when KB is more functional
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class KnowledgeWeaver:
    def __init__(self, kb=None, initial_query:str=""): # kb: KnowledgeBase
        """
        Initializes the KnowledgeWeaver.
        kb: KnowledgeBase instance, for broader context, definitions, or relationships.
        initial_query: The initial user query that started the workflow, for context.
        """
        self.kb = kb # Not used extensively in this version
        self.initial_query = initial_query
        logger.info("KnowledgeWeaver initialized.")

    def _extract_keywords_from_insights(self, evidence_packages: list[dict], top_n_keywords=5) -> list[str]:
        """
        Extracts common keywords from contributions and methodologies of analyzed papers.
        This is a simple approach; more advanced would use TF-IDF or actual NLP keyword extraction.
        """
        all_terms = []
        for pkg in evidence_packages:
            analysis_summary = pkg.get("original_analysis_summary", {})

            # Add methodologies
            all_terms.extend([m.lower() for m in analysis_summary.get("methodologies", [])])

            # Add terms from contributions (simple split for now)
            for contrib in analysis_summary.get("contributions", []):
                all_terms.extend([word.lower() for word in contrib.split() if len(word) > 3]) # Basic filter

        if not all_terms:
            return []

        # Get most common terms
        common_terms = [term for term, count in Counter(all_terms).most_common(top_n_keywords)]
        return common_terms

    def _group_and_theme_insights(self, evidence_packages: list[dict]) -> list[dict]:
        """
        Groups evidence packages into themes based on shared methodologies or keywords.
        This is a rule-based theming approach. More advanced would use clustering on embeddings.
        """
        logger.info(f"KnowledgeWeaver: Grouping {len(evidence_packages)} evidence packages into themes.")
        if not evidence_packages:
            return []

        # Theming strategy:
        # 1. Primary: Group by the first methodology listed if available and common.
        # 2. Secondary: If no clear methodology groups, try to find common keywords from claims/contributions.
        # For this version, let's simplify and use the primary methodology as the theme name.

        themes_map = defaultdict(lambda: {"theme_name": "", "summary_points": [], "papers_in_theme": [], "paper_titles_in_theme": [], "has_quantitative_data": False, "representative_keywords": []})

        for pkg in evidence_packages:
            paper_id = pkg.get("paper_id", "UnknownPaper")
            paper_title = pkg.get("title", "Untitled Paper")
            analysis_summary = pkg.get("original_analysis_summary", {})
            methodologies = analysis_summary.get("methodologies", [])

            # Use the first methodology as a theme candidate, or "General Research" if none.
            # A more robust approach would be to use all methodologies and see which ones form significant clusters.
            theme_candidate = methodologies[0] if methodologies else "General Research Insights"

            # Normalize theme names a bit (e.g. "CNN" and "Convolutional Neural Network" could be merged by a more advanced system)
            # For now, exact match.
            current_theme_map = themes_map[theme_candidate]
            current_theme_map["theme_name"] = theme_candidate # Set it if first time

            current_theme_map["papers_in_theme"].append(paper_id)
            current_theme_map["paper_titles_in_theme"].append(paper_title)

            # Add first contribution as a summary point for the theme
            contributions = analysis_summary.get("contributions", [])
            if contributions:
                 current_theme_map["summary_points"].append(f"Paper '{paper_title}' ({paper_id}) reports: {contributions[0]}")

            # Check for quantitative data (very basic check based on DeepAnalyzer's output)
            if "outperform" in analysis_summary.get("summary","").lower() or \
               "accuracy of" in analysis_summary.get("summary","").lower() or \
               analysis_summary.get("experimental_results_summary"): # If DeepAnalyzer provided one
                 current_theme_map["has_quantitative_data"] = True

            # Collect keywords for the theme (e.g. from claims or methodologies)
            # For now, just add the paper's own methodologies to the theme's keywords
            current_theme_map["representative_keywords"].extend(methodologies)


        # Post-process themes
        final_themes = []
        for theme_name, data in themes_map.items():
            # Create a concise summary for the theme
            theme_summary = f"This theme focuses on {theme_name}, covering {len(data['papers_in_theme'])} paper(s). "
            if data["summary_points"]:
                theme_summary += "Key aspects include: " + "; ".join(data["summary_points"][:2]) # Max 2 points for brevity
                if len(data["summary_points"]) > 2: theme_summary += "..."
            else:
                theme_summary += "It explores various applications and findings related to this area."

            # Get unique, top N keywords for the theme
            if data["representative_keywords"]:
                top_theme_keywords = [kw for kw, count in Counter(data["representative_keywords"]).most_common(3)]
            else:
                top_theme_keywords = []

            final_themes.append({
                "theme_name": theme_name,
                "theme_summary": theme_summary,
                "papers_in_theme": list(set(data["papers_in_theme"])), # Unique paper IDs
                "paper_titles_in_theme": list(set(data["paper_titles_in_theme"])),
                "has_quantitative_data": data["has_quantitative_data"],
                "representative_keywords": top_theme_keywords
            })

        if not final_themes and evidence_packages: # Fallback if no themes created (e.g. all "General Research")
            logger.warning("Could not form distinct themes; creating a general summary theme.")
            all_paper_ids = [pkg.get("paper_id") for pkg in evidence_packages]
            all_paper_titles = [pkg.get("title") for pkg in evidence_packages]
            # Basic check if any package indicates quantitative data
            has_quant_data_overall = any(pkg.get("original_analysis_summary",{}).get("experimental_results_summary","") for pkg in evidence_packages)

            return [{
                "theme_name": "Overall Summary of Findings",
                "theme_summary": f"Aggregated insights from {len(evidence_packages)} analyzed papers.",
                "papers_in_theme": all_paper_ids,
                "paper_titles_in_theme": all_paper_titles,
                "has_quantitative_data": has_quant_data_overall,
                "representative_keywords": self._extract_keywords_from_insights(evidence_packages, 3)
            }]

        logger.info(f"Formed {len(final_themes)} themes.")
        return final_themes


    def generate_report(self, evidence_packages: list[dict], query_context:str = None) -> dict:
        """
        Integrates and synthesizes knowledge from various evidence packages.
        evidence_packages: A list of evidence packages from EvidenceBuilder.
        query_context: The original query or refined query for context.
        Returns a structured dictionary ready for the WritingEngine.
        """
        logger.info(f"KnowledgeWeaver: Generating integrated report from {len(evidence_packages)} evidence packages.")

        # Use provided query_context or fall back to the one from init
        current_query = query_context if query_context else self.initial_query

        if not evidence_packages:
            logger.warning("No evidence packages provided to KnowledgeWeaver. Returning empty report structure.")
            return {
                "overall_topic": current_query if current_query else "Not Specified",
                "introduction_text": "No analyzed insights were available to generate a report.",
                "themed_sections": [],
                "methodology_overview": [],
                "reproducibility_summary_data": {"checked_count":0, "reproduced_count":0, "details":[]},
                "conclusion_text": "Unable to draw conclusions due to lack of input.",
                "source_evidence_packages": [], # Pass through the (empty) list
                "statistics": {"total_papers_analyzed": 0, "total_claims_processed":0}
            }

        # 1. Group insights into themes
        themed_sections = self._group_and_theme_insights(evidence_packages)

        # 2. Synthesize introduction and conclusion (placeholders for now)
        # LLM integration point for better summarization.
        overall_topic = f"Literature Survey on: {current_query}" if current_query else \
                        f"Literature Survey based on {len(evidence_packages)} Analyzed Papers"

        common_keywords = self._extract_keywords_from_insights(evidence_packages)
        intro_keywords_str = f"Key themes explored include {', '.join(common_keywords)}." if common_keywords else ""

        introduction_text = (f"This report synthesizes findings from {len(evidence_packages)} key academic papers "
                             f"related to '{current_query if current_query else 'the specified research area'}'. "
                             f"It covers {len(themed_sections)} primary themes and discusses various methodologies "
                             f"identified in the literature. {intro_keywords_str}")

        conclusion_text = (f"The analyzed literature indicates active research across {len(themed_sections)} identified themes. "
                           f"Common methodologies such as {', '.join(self._extract_all_methodologies(evidence_packages)[:3]) if self._extract_all_methodologies(evidence_packages) else 'various approaches'} are prevalent. "
                           "Further investigation into cross-theme connections and emerging trends is warranted. "
                           "Reproducibility remains a key aspect for future validation.")


        # 3. Aggregate common methodologies and reproducibility data
        all_methodologies_list = self._extract_all_methodologies(evidence_packages)
        methodology_counts = Counter(all_methodologies_list)
        methodology_overview = [{"method": m, "count": c} for m, c in methodology_counts.most_common(10)] # Top 10

        repro_summary = self._summarize_reproducibility(evidence_packages)

        total_claims_processed = sum(len(pkg.get("processed_claims", [])) for pkg in evidence_packages)

        # 4. Structure the output for WritingEngine
        structured_report_data = {
            "overall_topic": overall_topic,
            "introduction_text": introduction_text,
            "themed_sections": themed_sections, # List of themes, each with a summary & related papers
            "methodology_overview": methodology_overview,
            "reproducibility_summary_data": repro_summary,
            "conclusion_text": conclusion_text,
            "source_evidence_packages": evidence_packages, # Pass through for detailed evidence
            "statistics": {
                "total_papers_analyzed": len(evidence_packages),
                "total_claims_processed": total_claims_processed,
                "total_themes_identified": len(themed_sections)
            }
            # "expected_themes_count" was for Validator, might remove or make it dynamic
        }

        logger.info("KnowledgeWeaver: Integrated report structure generated.")
        return structured_report_data

    def _extract_all_methodologies(self, evidence_packages: list[dict]) -> list[str]:
        """Helper to get a flat list of all methodologies mentioned."""
        all_methods = []
        for pkg in evidence_packages:
            methods = pkg.get("original_analysis_summary", {}).get("methodologies", [])
            all_methods.extend(methods)
        return all_methods

    def _summarize_reproducibility(self, evidence_packages: list[dict]) -> dict:
        """Helper to summarize reproducibility information."""
        checked_count = 0
        reproduced_count = 0
        details = []
        for pkg in evidence_packages:
            repro_assess = pkg.get("reproducibility_assessment", {})
            if repro_assess.get("reproducible_check_attempted"):
                checked_count += 1
                if repro_assess.get("reproducible"):
                    reproduced_count += 1
                details.append({
                    "paper_id": pkg.get("paper_id"),
                    "title": pkg.get("title"),
                    "checked": True,
                    "reproduced": repro_assess.get("reproducible"),
                    "success_rate": repro_assess.get("success_rate"),
                    "url": repro_assess.get("checked_url")
                })
            # Optionally, add papers for which check was not attempted
            # else:
            #     details.append({"paper_id": pkg.get("paper_id"), "title": pkg.get("title"), "checked": False})

        return {
            "total_papers_checked_for_reproducibility": checked_count,
            "total_papers_reproduced_successfully": reproduced_count,
            "details": details # List of dicts with per-paper reproducibility info
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Mock KB for testing
    # class MockKB_ForKW:
    #     def __init__(self): logger.info("MockKB_ForKW initialized.")
    # mock_kb = MockKB_ForKW()

    weaver = KnowledgeWeaver(kb=None, initial_query="AI in Healthcare") # No actual KB needed for this version

    # Sample data from EvidenceBuilder (list of evidence packages)
    # Using the structure from the updated EvidenceBuilder
    mock_evidence_packages = [
        { # Paper 1
            "paper_id": "arxiv:001", "title": "Paper on CNNs for X-Ray Analysis",
            "url": "http://...",
            "original_analysis_summary": {
                "summary": "This paper uses CNNs for X-Ray analysis.",
                "contributions": ["Novel CNN architecture.", "90% accuracy on ChestXRay dataset."],
                "limitations": ["Needs more diverse data."],
                "methodologies": ["CNN", "Deep Learning", "Image Classification"]
            },
            "processed_claims": [{"text": "CNNs are effective.", "verification_status": "Verified"}],
            "reproducibility_assessment": {"reproducible_check_attempted":True, "reproducible": True, "success_rate": 0.9, "checked_url": "git://cnn_xray"}
        },
        { # Paper 2
            "paper_id": "pubmed:002", "title": "NLP for Patient Record Summarization",
            "url": "http://...",
            "original_analysis_summary": {
                "summary": "Transformers for summarizing patient records.",
                "contributions": ["New summarization technique using BERT.", "Reduces physician workload."],
                "limitations": ["Requires large datasets for BERT fine-tuning."],
                "methodologies": ["NLP", "Transformer", "BERT", "Text Summarization"]
            },
            "processed_claims": [{"text": "BERT model improves summarization.", "verification_status": "Verified"}],
            "reproducibility_assessment": {"reproducible_check_attempted":True, "reproducible": False, "success_rate": 0.4, "checked_url": "git://bert_summarizer"}
        },
        { # Paper 3
            "paper_id": "arxiv:003", "title": "Advanced CNNs for Medical Imaging",
            "url": "http://...",
            "original_analysis_summary": {
                "summary": "Further explores CNNs in medical imaging.",
                "contributions": ["EfficientNet variant for faster processing.", "Comparable accuracy to larger models."],
                "limitations": ["Tested only on one imaging modality."],
                "methodologies": ["CNN", "Deep Learning", "EfficientNet"]
            },
            "processed_claims": [{"text": "EfficientNet variant is faster.", "verification_status": "Verified"}],
            "reproducibility_assessment": {"reproducible_check_attempted":False, "log":"No code link found."} # Not checked
        }
    ]

    report_structure = weaver.generate_report(mock_evidence_packages, query_context="Applications of AI in Medical Diagnostics")

    print("\n--- KnowledgeWeaver Output Structure ---")
    import json
    print(json.dumps(report_structure, indent=2, ensure_ascii=False)) # ensure_ascii=False for readability if non-ASCII
    print("--- End of Output ---")
