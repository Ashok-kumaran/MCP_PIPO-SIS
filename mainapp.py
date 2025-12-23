import os
from dotenv import load_dotenv
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from yaspin import yaspin
from yaspin.spinners import Spinners
import asyncio
from clientdemo import run_with_prompt

load_dotenv()

MCP_SERVER_PATH = r"C:\Users\PremkumarUthayakumar\Desktop\projects\SAPIS_Stdio\dist\index.js"

# ==========================================================
# 1. LLM (SAP AI CORE)
# ==========================================================

deployment_id = os.getenv("LLM_DEPLOYMENT_ID")

llm = ChatOpenAI(
    deployment_id=deployment_id,
    temperature=0,
    streaming=True,
)

# ==========================================================
# 2. PROMPT – HUMAN UNDERSTANDING (HITL)
# ==========================================================

understanding_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an SAP CPI integration architect.

Explain in CLEAR BULLET POINTS what the iFlow will do.

Format STRICTLY like this:

• Sender Adapter:
• Receiver Adapter:
• Business Purpose:
  - ...
• Main Processing Steps:
  - ...
• Error Handling:
  - ...
• Assumptions:
  - ...

Rules:
- Use SAP CPI terminology
- Be precise and concise
- Do NOT invent adapters
- Do NOT output JSON or XML
- This output is for HUMAN review
- If the user doesn't specify iflow name and package name ask for one.
- If the user asks for details of iflow, just confirm the iflow name alone, by asking questions, don't ask for iflow details.
- If the user asks for details of package, just confirm the package name alone, by asking questions, don't ask for package details.
        """
    ),  
    ("human", "{user_query}")
])

understanding_chain = understanding_prompt | llm

# ==========================================================
# 3. PROMPT – HITL REFINEMENT
# ==========================================================

refine_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an SAP CPI integration architect.

You will be given:
1) The current iFlow understanding
2) User feedback describing changes

Update the understanding accordingly.

Rules:
- Apply ONLY the requested changes
- Keep the same bullet structure
- Do NOT invent new functionality
- Do NOT remove anything unless explicitly asked
- Return the FULL updated understanding
        """
    ),
    ("human", "Current understanding:\n{current_summary}"),
    ("human", "User feedback:\n{feedback}")
])

refine_chain = refine_prompt | llm

# ==========================================================
# 4. PROMPT – EXECUTION INSTRUCTION (CRITICAL FIX)
# ==========================================================

execution_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an SAP CPI technical planner.

From the APPROVED iFlow understanding, Send the approved summary.

Rules:
- if the user asks for details of iflow, just provide details of iflow alone.
- if the user asks for details of package, just provide details of package alone.
        """
    ),
    ("human", "{approved_summary}")
])

execution_chain = execution_prompt | llm

# ==========================================================
# 5. HUMAN-IN-THE-LOOP
# ==========================================================

def human_in_the_loop(summary: str) -> str | None:
    while True:
        print("\n========== IFLOW UNDERSTANDING ==========\n")
        print(summary)
        print("\n========================================")

        decision = input(
            "\nIs this what you intended? (yes / edit / no): "
        ).strip().lower()

        if decision == "yes":
            return summary

        if decision == "no":
            return None

        if decision == "edit":
            feedback = input(
                "\nDescribe what you want to change (plain English):\n\n"
            )

            with yaspin(
                Spinners.dots,
                text="Updating understanding based on your feedback...",
                color="cyan"
            ) as spinner:
                summary = refine_chain.invoke({
                    "current_summary": summary,
                    "feedback": feedback
                }).content
                spinner.ok("✅")

# ==========================================================
# 6. MAIN
# ==========================================================

def main():

    print("\n🧩 SAP CPI iFlow Understanding (Human-in-the-Loop)\n")

    user_query = input(
        "Describe the iFlow you want to create (2–3 lines):\n\n"
    )

    with yaspin(
        Spinners.dots,
        text="Understanding your iFlow request...",
        color="cyan"
    ) as spinner:
        summary = understanding_chain.invoke(
            {"user_query": user_query}
        ).content
        spinner.ok("✅")

    final_summary = human_in_the_loop(summary)

    if not final_summary:
        print("\n❌ iFlow creation stopped. Please rephrase your request.")
        return

    print("\n✅ Understanding confirmed.")
    print("\n➡️ FINAL APPROVED IFLOW SUMMARY:\n")
    print(final_summary)

    # ======================================================
    # 🔑 EXECUTION INSTRUCTION (NEW, IMPORTANT)
    # ======================================================

    with yaspin(
        Spinners.dots,
        text="Preparing execution-safe instruction...",
        color="yellow"
    ) as spinner:
        execution_instruction = execution_chain.invoke({
            "approved_summary": final_summary
        }).content
        spinner.ok("✅")

    print("\n➡️ EXECUTION INSTRUCTION (SENT TO MCP):\n")
    print(execution_instruction)

    # ======================================================
    # MCP EXECUTION
    # ======================================================

    print("\n➡️ Executing MCP...\n")

    result = asyncio.run(
        run_with_prompt(MCP_SERVER_PATH, execution_instruction)
    )

    print("\n🚀 MCP RESULT:\n")
    print(result)

if __name__ == "__main__":
    main()
