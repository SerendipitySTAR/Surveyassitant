from .control_center.control_center import ControlCenter
from .knowledge_base.knowledge_base import KnowledgeBase
from .literature_discovery.literature_hunter import LiteratureHunter # Placeholder
from .deep_analysis.deep_analyzer import DeepAnalyzer # Placeholder
from .evidence_builder.evidence_builder import EvidenceBuilder # Placeholder
from .knowledge_weaver.knowledge_weaver import KnowledgeWeaver
from .writing_engine.writing_engine import WritingEngine
from .validator.validator import Validator
from .utils import ensure_dir_exists, load_config # Added ensure_dir_exists
import os
import json # For saving JSON validation report
import logging # For logging file save operations
from datetime import datetime # Added import

logger = logging.getLogger(__name__)


def format_output(main_report_data: dict, # This is now the structured data from KW
                  validation_report: dict,
                  query: str,
                  writing_engine: WritingEngine) -> dict:
    """
    Formats the final output package, generating Markdown, LaTeX, and simulating DOCX/PDF.
    Saves outputs to files based on README structure.

    Args:
        main_report_data (dict): Structured data from KnowledgeWeaver.
        validation_report (dict): Report from Validator.
        query (str): The original query for naming.
        writing_engine (WritingEngine): Instance of the WritingEngine.

    Returns:
        A dictionary summarizing the output paths and messages.
    """
    logger.info("Formatting final output package...")

    config = load_config()
    base_output_dir = config.get("output_dir", "output")

    # Sanitize query for use in filenames
    safe_query_name = "".join(c if c.isalnum() else "_" for c in query[:50]) # Limit length
    timestamp_str = validation_report.get('evaluation_timestamp', datetime.utcnow().isoformat()).split('T')[0]

    # Simplified output for testing sandbox limits
    simple_output_dir = os.path.join(base_output_dir, f"survey_run_{safe_query_name}_{timestamp_str}")
    ensure_dir_exists(simple_output_dir)

    output_summary = {
        "package_directory": simple_output_dir,
        "files_generated": [],
        "messages": []
    }

    # 1. Generate and save Markdown report (simplified path)
    try:
        md_content = writing_engine.compose(main_report_data, output_format="markdown")
        md_filepath = os.path.join(simple_output_dir, f"{safe_query_name}_survey.md")
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(md_content)
        output_summary["files_generated"].append({"format": "Markdown", "path": md_filepath})
        logger.info(f"Markdown report saved to: {md_filepath}")

        # Simulate DOCX by saving the MD file that would be converted
        docx_sim_md_path = os.path.join(simple_output_dir, f"{safe_query_name}_for_docx.md")
        docx_sim_result = writing_engine.to_docx_simulation(md_content, docx_sim_md_path)
        output_summary["messages"].append(docx_sim_result["message"])
        if docx_sim_result.get("success"):
             output_summary["files_generated"].append({"format": "DOCX_Simulation_MD", "path": docx_sim_result["markdown_filepath"]})

    except Exception as e:
        logger.error(f"Error generating Markdown or simulating DOCX: {e}")
        output_summary["messages"].append(f"Error (MD/DOCX): {e}")

    # 2. Generate and save LaTeX report (simplified path)
    try:
        latex_content = writing_engine.compose(main_report_data, output_format="latex")
        tex_filepath = os.path.join(simple_output_dir, f"{safe_query_name}_survey.tex")
        with open(tex_filepath, "w", encoding="utf-8") as f:
            f.write(latex_content)
        output_summary["files_generated"].append({"format": "LaTeX", "path": tex_filepath})
        logger.info(f"LaTeX report saved to: {tex_filepath}")

        pdf_sim_result = writing_engine.to_pdf_simulation(latex_content, tex_filepath)
        output_summary["messages"].append(pdf_sim_result["message"])

    except Exception as e:
        logger.error(f"Error generating LaTeX or simulating PDF: {e}")
        output_summary["messages"].append(f"Error (LaTeX/PDF): {e}")

    # 3. Save Validation Report (simplified path)
    try:
        val_report_filepath = os.path.join(simple_output_dir, f"{safe_query_name}_validation_report.json")
        with open(val_report_filepath, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        output_summary["files_generated"].append({"format": "ValidationReport_JSON", "path": val_report_filepath})
        logger.info(f"Validation report saved to: {val_report_filepath}")
    except Exception as e:
        logger.error(f"Error saving validation report: {e}")
        output_summary["messages"].append(f"Error (Validation Report): {e}")

    return output_summary


def select_priority_papers(papers: list, criteria: dict = None) -> list:
    """
    Selects priority papers for deep analysis.
    Placeholder: returns first few papers or all if less than a certain number.
    """
    if not papers:
        return []
    # In a real system, this would involve more sophisticated ranking/selection
    # based on relevance, recency, citation count, etc.
    limit = criteria.get("priority_paper_limit", 5) if criteria else 5
    print(f"Selecting up to {limit} priority papers from {len(papers)} total.")
    return papers[:limit]


def enhanced_workflow(query: str, max_iterations=3):
    """
    Implements the enhanced workflow for generating a literature survey.
    This is the main orchestrator function.
    """
    print(f"\n--- Starting Enhanced Workflow for Query: '{query}' ---")

    # Initialization
    control = ControlCenter()
    kb = KnowledgeBase() # Initialize with default or configured embedding model

    # Placeholder for overall progress tracking
    workflow_iteration = 1

    # Main loop for iterative refinement (as suggested by README)
    while workflow_iteration <= max_iterations:
        print(f"\n--- Iteration {workflow_iteration} ---")

        # Phase 1: Dynamic Planning (already part of ControlCenter initialization conceptually)
        roadmap = control.generate_roadmap(query) # Generates/updates roadmap for the query
        current_mode = control.dynamic_scheduling() # Determine current operating mode
        print(f"Operating in mode: {current_mode}")

        # Phase 2: Literature Discovery
        print("\nPhase: Literature Discovery")
        # In a real system, LiteratureHunter might take roadmap specifics or control signals
        literature_hunter = LiteratureHunter()
        # The query to fetch_papers might be refined by the control center based on the roadmap
        discovered_papers = literature_hunter.fetch_papers(query, roadmap.get("phases")[0].get("KPI"))
        if not discovered_papers:
            print("No papers discovered. Workflow cannot continue effectively.")
            # control.emergency_plan("NO_PAPERS_FOUND") # Example emergency
            return {"status": "failed", "reason": "No papers discovered", "query": query}

        kb.incremental_update(discovered_papers) # Update knowledge base with new findings

        # Phase 3: Deep Analysis & Evidence Building
        print("\nPhase: Deep Analysis & Evidence Building")
        # Prioritize papers for deep analysis (e.g., based on relevance, roadmap KPIs)
        # The roadmap's "Deep Analysis" phase might have criteria for paper selection
        priority_papers_list = select_priority_papers(discovered_papers, criteria={"priority_paper_limit": 5})

        analyzed_insights = []
        if not priority_papers_list:
            print("No priority papers selected for deep analysis.")
        else:
            for paper_summary in priority_papers_list: # Assuming fetch_papers returns list of summaries/IDs
                print(f"  Analyzing paper: {paper_summary.get('id', 'Unknown ID')} - {paper_summary.get('title', 'N/A')}")
                # DeepAnalyzer might need the full paper content, or use kb to fetch it
                analyzer = DeepAnalyzer(paper_summary) # Pass paper data or ID
                analysis_result = analyzer.run() # Returns structured info, claims, etc.

                # EvidenceBuilder takes the analysis result to build/verify evidence
                # It might interact with the KnowledgeBase or external tools (CodeSandbox)
                evidence_builder = EvidenceBuilder(kb) # Pass kb for context
                evidence_package = evidence_builder.build_and_verify(analysis_result, paper_summary)
                analyzed_insights.append(evidence_package)

        if not analyzed_insights:
            print("No insights generated from deep analysis. Reviewing literature or analysis strategy.")
            # control.emergency_plan("NO_INSIGHTS_GENERATED")
            # Potentially loop back or adjust strategy if iteration allows
            if workflow_iteration < max_iterations:
                print("Attempting to adjust strategy (simulated) and re-iterate.")
                query += " (refined search)" # Simplistic refinement
                workflow_iteration += 1
                continue
            else:
                return {"status": "failed", "reason": "No insights generated after deep analysis", "query": query}

        # Phase 4: Knowledge Integration/Weaving
        print("\nPhase: Knowledge Integration")
        # KnowledgeWeaver synthesizes a structured report/knowledge representation from insights
        knowledge_weaver = KnowledgeWeaver(kb) # May use KB for broader context
        integrated_report_data = knowledge_weaver.generate_report(analyzed_insights)

        # Phase 5: Writing & Review
        print("\nPhase: Writing & Review")
        writing_engine = WritingEngine() # Could be configured with templates, style guides
        draft_document = writing_engine.compose(integrated_report_data) # Takes structured data, produces text

        validator = Validator(kb) # Validator might use KB for fact-checking against known literature
        validation_report = validator.evaluate(draft_document, integrated_report_data) # Evaluate draft and underlying data

        print(f"\nValidation Report (Iteration {workflow_iteration}):")
        print(f"  Overall Score: {validation_report.get('overall_score', 'N/A')}")
        print(f"  Fact Consistency: {validation_report.get('fact_consistency', 'N/A')}")
        print(f"  Plagiarism Score: {validation_report.get('plagiarism_score', 'N/A')}")

        # Phase 6: Closed-Loop Control & Iteration Decision
        # README suggests a quality threshold, e.g., 0.95
        quality_threshold = roadmap.get("phases")[-1].get("KPI_threshold", 0.95) # Assuming KPI is like "Credibility ≥0.95"

        if validation_report.get('overall_score', 0) < quality_threshold:
            print(f"Overall score {validation_report.get('overall_score', 0)} is below threshold {quality_threshold}.")
            if workflow_iteration < max_iterations:
                print("Attempting iterative refinement...")
                # control.adjust_plan(validation_report.get('weak_areas', [])) # TODO: Implement adjust_plan in ControlCenter
                # For now, simulate a simple adjustment like refining the query or focusing analysis
                query += " (iterative refinement)" # Simplistic query adjustment for next loop
                # Potentially, the control center could re-prioritize tasks or parameters for agents.
                workflow_iteration += 1
            else:
                print("Max iterations reached. Outputting current best effort.")
                # Pass integrated_report_data (from KW) to format_output, not just one rendered draft
                final_output_summary = format_output(integrated_report_data, validation_report, query, writing_engine)
                kb.close_databases()
                return {"status": "completed_max_iterations", "output_summary": final_output_summary, "query": query}
        else:
            print(f"Quality threshold met or exceeded. Overall score: {validation_report.get('overall_score', 0)}")
            final_output_summary = format_output(integrated_report_data, validation_report, query, writing_engine)
            kb.close_databases()
            return {"status": "completed_successfully", "output_summary": final_output_summary, "query": query}

    # Fallback if loop finishes due to max_iterations without explicit success
    # This part should ideally be covered by the logic within the loop's final iteration.
    # If loop finishes, it means the last iteration's draft_document and validation_report are the final ones.
    logger.info("Max iterations reached by falling through loop. Workflow finished.")
    kb.close_databases()

    # Check if final data for formatting exists (it should if loop ran at least once)
    if 'integrated_report_data' in locals() and 'validation_report' in locals() and 'writing_engine' in locals():
        final_output_summary = format_output(integrated_report_data, validation_report, query, writing_engine)
        return {"status": "completed_max_iterations_fallback", "output_summary": final_output_summary, "query": query}
    else: # Should not happen if workflow ran through at least one paper analysis cycle
        logger.error("Workflow ended unexpectedly without producing necessary data for final output formatting.")
        return {"status": "failed_unexpected_finish", "reason": "Workflow ended without producing a final draft or necessary data.", "query": query}


if __name__ == '__main__':
    print("Starting standalone test of enhanced_workflow...")

    # Test scenario 1: Basic run that should "succeed" with placeholder logic
    initial_query = "Recent advancements in renewable energy"
    result = enhanced_workflow(initial_query, max_iterations=2)
    print(f"\n--- Workflow Result for '{initial_query}' ---")
    print(f"Status: {result.get('status')}")
    if result.get('output'):
        print(f"Output (summary): {result['output']}")

    print("\n---------------------------------------------------\n")

    # Test scenario 2: Simulate a case where validation might initially fail (if logic was real)
    # For placeholders, this will likely behave similarly unless we force different scores.
    # To make this more illustrative with placeholders, one might modify placeholder agents
    # to return different results based on the query or iteration count.
    # For now, it will just run through.
    # initial_query_2 = "Challenges in AI ethics"
    # result_2 = enhanced_workflow(initial_query_2, max_iterations=3)
    # print(f"\n--- Workflow Result for '{initial_query_2}' ---")
    # print(f"Status: {result_2.get('status')}")
    # if result_2.get('output'):
    #     print(f"Output (summary): {result_2['output']}")

    print("\nStandalone test of enhanced_workflow finished.")
