from agents import Runner

from research_agents.manager import manager_agent
from research_agents.researcher import research_agent
from research_agents.analyst import analysis_agent
from research_agents.writer import writer_agent
from research_agents.reviewer import reviewer_agent

from models.schemas import (
    ResearchPlan,
    ResearchResult,
    AnalysisResult,
)


async def run_manager(topic: str) -> ResearchPlan:
    result = await Runner.run(
        manager_agent,
        topic,
    )

    return result.final_output


async def run_research(plan: ResearchPlan) -> ResearchResult:
    prompt = f"""
    Research the following topic.

    TOPIC:
    {plan.topic}

    OBJECTIVE:
    {plan.objective}

    RESEARCH TASKS:
    """

    for i, task in enumerate(plan.research_tasks, start=1):
        prompt += f"\n{i}. {task}"

    result = await Runner.run(
        research_agent,
        prompt,
    )

    return result.final_output


async def run_analysis(
        research: ResearchResult,
) -> AnalysisResult:

    prompt = f"""
    Analyze the following research.

    TOPIC:
    {research.topic}

    RESEARCH FINDINGS:
    """

    for i, finding in enumerate(
            research.findings,
            start=1,
    ):
        prompt += f"""

        FINDING {i}

        Title:
        {finding.title}

        Summary:
        {finding.summary}

        Source:
        {finding.source_url}
        """

    if research.gaps:
        prompt += "\n\nRESEARCH GAPS:\n"

        for gap in research.gaps:
            prompt += f"- {gap}\n"

    result = await Runner.run(
        analysis_agent,
        prompt,
    )

    return result.final_output


async def run_writer(
        research: ResearchResult,
        analysis: AnalysisResult,
) -> str:

    prompt = f"""
    Write a report using the following information.

    TOPIC:
    {research.topic}

    RESEARCH:
    """

    for i, finding in enumerate(
            research.findings,
            start=1,
    ):
        prompt += f"""

        FINDING {i}

        Title:
        {finding.title}

        Summary:
        {finding.summary}

        Source:
        {finding.source_url}
        """

    prompt += """

    ANALYSIS:

    KEY FINDINGS:
    """

    for finding in analysis.key_findings:
        prompt += f"\n- {finding}"

    prompt += "\n\nCOMPARISONS:\n"

    for comparison in analysis.comparisons:
        prompt += f"\n- {comparison}"

    prompt += f"""

    RECOMMENDATION:
    {analysis.recommendation}

    MISSING INFORMATION:
    """

    for item in analysis.missing_information:
        prompt += f"\n- {item}"

    result = await Runner.run(
        writer_agent,
        prompt,
    )

    return result.final_output


async def run_reviewer(
        topic: str,
        research: ResearchResult,
        analysis: AnalysisResult,
        report: str,
):
    prompt = f"""
    Review this research report.

    ORIGINAL USER QUESTION:
    {topic}

    RESEARCH:
    """

    for i, finding in enumerate(
            research.findings,
            start=1,
    ):
        prompt += f"""

        FINDING {i}

        Title:
        {finding.title}

        Summary:
        {finding.summary}

        Source:
        {finding.source_url}
        """

    prompt += """

    ANALYSIS:

    KEY FINDINGS:
    """

    for finding in analysis.key_findings:
        prompt += f"\n- {finding}"

    prompt += "\n\nCOMPARISONS:\n"

    for comparison in analysis.comparisons:
        prompt += f"\n- {comparison}"

    prompt += f"""

    RECOMMENDATION:
    {analysis.recommendation}

    REPORT:
    {report}
    """

    result = await Runner.run(
        reviewer_agent,
        prompt,
    )

    return result.final_output


async def revise_report(
        research: ResearchResult,
        analysis: AnalysisResult,
        previous_report: str,
        feedback: list[str],
) -> str:

    prompt = f"""
    Revise the research report below.

    Fix the issues identified by the reviewer.

    Do NOT perform additional research.
    Do NOT invent facts.

    RESEARCH:
    """

    for finding in research.findings:
        prompt += f"""

        Title:
        {finding.title}

        Summary:
        {finding.summary}

        Source:
        {finding.source_url}
        """

    prompt += """

    ANALYSIS:
    """

    for finding in analysis.key_findings:
        prompt += f"\n- {finding}"

    prompt += f"""

    RECOMMENDATION:
    {analysis.recommendation}

    PREVIOUS REPORT:
    {previous_report}

    REVIEWER FEEDBACK:
    """

    for item in feedback:
        prompt += f"\n- {item}"

    result = await Runner.run(
        writer_agent,
        prompt,
    )

    return result.final_output


# =========================================
# FULL WORKFLOW
# =========================================

async def run_full_research_team(
        topic: str,
        max_revisions: int = 1,
        progress_callback=None,
):

    # Manager
    if progress_callback:
        progress_callback("running", "🧠 Manager is creating the research plan...")

    plan = await run_manager(topic)

    if progress_callback:
        progress_callback("done", "✅ Manager completed the research plan")


    # Researcher
    if progress_callback:
        progress_callback("running", "🔎 Researcher is searching the web...")

    research = await run_research(plan)

    if progress_callback:
        progress_callback("done", "✅ Researcher completed the research")


    # Analyst
    if progress_callback:
        progress_callback("running", "📊 Analyst is analyzing the findings...")

    analysis = await run_analysis(research)

    if progress_callback:
        progress_callback("done", "✅ Analyst completed the analysis")


    # Writer
    if progress_callback:
        progress_callback("running", "✍️ Writer is preparing the report...")

    current_report = await run_writer(
        research,
        analysis,
    )

    if progress_callback:
        progress_callback("done", "✅ Writer completed the first draft")


    # Reviewer
    review = None
    revision_count = 0

    for revision in range(max_revisions):

        if progress_callback:
            progress_callback(
                "running",
                f"🧐 Reviewer is checking the report — round {revision + 1}..."
            )

        review = await run_reviewer(
            topic,
            research,
            analysis,
            current_report,
        )

        if review.approved:

            if progress_callback:
                progress_callback(
                    "done",
                    f"✅ Reviewer approved the report — Score: {review.score}/10"
                )

            break


        if progress_callback:
            progress_callback(
                "warning",
                f"⚠️ Reviewer requested changes — Score: {review.score}/10"
            )


        revision_count += 1

        if progress_callback:
            progress_callback(
                "running",
                f"✍️ Writer is revising the report — revision {revision_count}..."
            )

        current_report = await revise_report(
            research,
            analysis,
            current_report,
            review.feedback,
        )

        if progress_callback:
            progress_callback(
                "done",
                f"✅ Revision {revision_count} completed"
            )


    return {
        "topic": topic,
        "plan": plan,
        "research": research,
        "analysis": analysis,
        "report": current_report,
        "review": review,
        "revision_count": revision_count,
    }
