import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

from services.workflow import run_full_research_team


# ----------------------------------
# Environment
# ----------------------------------

load_dotenv()


# ----------------------------------
# Page configuration
# ----------------------------------

st.set_page_config(
    page_title="AI Research Team",
    page_icon="🤖",
    layout="wide",
)


# ----------------------------------
# API key check
# ----------------------------------

if not os.getenv("OPENAI_API_KEY"):

    st.error(
        "OPENAI_API_KEY was not found. "
        "Add it to your .env file."
    )

    st.stop()


# ----------------------------------
# Session state
# ----------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None


# ----------------------------------
# Header
# ----------------------------------

st.title("🤖 AI Research Team")

st.caption(
    "Manager → Researcher → Analyst → Writer → Reviewer"
)

st.write(
    "Enter a research question. Multiple AI agents will "
    "plan, research, analyze, write, and review a report."
)


# ----------------------------------
# Input
# ----------------------------------

with st.form(
        "research_form",
        clear_on_submit=True,
):

    question = st.text_area(
        "Research Question",
        placeholder=(
            "Example: Compare RAG and fine-tuning "
            "for enterprise AI applications."
        ),
        height=120,
    )

    submitted = st.form_submit_button(
        "🚀 Run Research Team",
        type="primary",
        use_container_width=True,
    )


# ----------------------------------
# Execute workflow
# ----------------------------------

if submitted:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a research question."
        )

    else:

        st.session_state.last_question = question

        with st.status(
                "🚀 Running AI research team...",
                expanded=True,
        ) as status:

            progress_bar = st.progress(
                0,
                text="Starting workflow..."
            )

            completed_steps = 0

            total_steps = 5

            def update_progress(state, message):

                global completed_steps

                if state == "running":

                    st.write(message)

                elif state == "done":

                    st.write(message)

                    completed_steps += 1

                    percent = min(
                        completed_steps / total_steps,
                        1.0
                    )

                    progress_bar.progress(
                        percent,
                        text=message
                    )

                elif state == "warning":

                    st.warning(message)


            try:

                result = asyncio.run(
                    run_full_research_team(
                        topic=question,
                        max_revisions=3,
                        progress_callback=update_progress,
                    )
                )

                st.session_state.result = result

                progress_bar.progress(
                    1.0,
                    text="✅ Research completed"
                )

                status.update(
                    label="✅ Research completed",
                    state="complete",
                    expanded=False,
                )

            except Exception as e:

                status.update(
                    label="❌ Workflow failed",
                    state="error",
                    expanded=True,
                )

                st.exception(e)

                st.session_state.result = result

                status.update(
                    label="✅ Research completed",
                    state="complete",
                    expanded=False,
                )

            except Exception as e:

                status.update(
                    label="❌ Workflow failed",
                    state="error",
                    expanded=True,
                )

                st.exception(e)


# ----------------------------------
# Results
# ----------------------------------

if st.session_state.result is not None:

    result = st.session_state.result

    st.divider()

    st.subheader("Research Question")

    st.info(
        st.session_state.last_question
    )


    final_tab, plan_tab, research_tab, analysis_tab, review_tab = st.tabs(
        [
            "📄 Final Report",
            "🧠 Plan",
            "🔎 Research",
            "📊 Analysis",
            "🧐 Review",
        ]
    )


    # ==================================
    # Final Report
    # ==================================

    with final_tab:

        st.markdown(
            result["report"]
        )

        st.divider()

        st.download_button(
            label="⬇️ Download Markdown Report",
            data=result["report"],
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )


    # ==================================
    # Research Plan
    # ==================================

    with plan_tab:

        plan = result["plan"]

        st.subheader("Objective")

        st.write(
            plan.objective
        )

        st.subheader("Research Tasks")

        for i, task in enumerate(
                plan.research_tasks,
                start=1,
        ):

            st.write(
                f"{i}. {task}"
            )


    # ==================================
    # Research
    # ==================================

    with research_tab:

        research = result["research"]

        st.subheader(
            "Research Findings"
        )

        for i, finding in enumerate(
                research.findings,
                start=1,
        ):

            with st.expander(
                    f"{i}. {finding.title}"
            ):

                st.write(
                    finding.summary
                )

                st.markdown(
                    f"**Source:** {finding.source_url}"
                )

        if research.gaps:

            st.subheader(
                "Research Gaps"
            )

            for gap in research.gaps:

                st.write(
                    f"- {gap}"
                )


    # ==================================
    # Analysis
    # ==================================

    with analysis_tab:

        analysis = result["analysis"]

        st.subheader(
            "Key Findings"
        )

        for finding in analysis.key_findings:

            st.write(
                f"- {finding}"
            )


        st.subheader(
            "Comparisons"
        )

        for comparison in analysis.comparisons:

            st.write(
                f"- {comparison}"
            )


        st.subheader(
            "Recommendation"
        )

        st.write(
            analysis.recommendation
        )


        if analysis.missing_information:

            st.subheader(
                "Missing Information"
            )

            for item in analysis.missing_information:

                st.write(
                    f"- {item}"
                )


    # ==================================
    # Reviewer
    # ==================================

    with review_tab:

        review = result["review"]

        if review is None:

            st.info(
                "No review was performed."
            )

        else:

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Score",
                f"{review.score}/10",
            )

            col2.metric(
                "Revisions",
                result["revision_count"],
            )

            if review.approved:

                col3.success(
                    "✅ Approved"
                )

            else:

                col3.warning(
                    "⚠️ Not Approved"
                )


            st.subheader(
                "Reviewer Feedback"
            )

            if review.feedback:

                for feedback in review.feedback:

                    st.write(
                        f"- {feedback}"
                    )

            else:

                st.success(
                    "No additional feedback."
                )
