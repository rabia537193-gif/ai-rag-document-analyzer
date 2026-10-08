import os
from langchain_core.prompts import PromptTemplate

# Mock LLM Function - Taake bina API key ke bhi aap test kar sakein
# Agar aapke paas OpenAI API key hai, to aap actual LLM use kar sakte hain
def mock_llm_response(prompt_text):
    """Simulates LLM response generation based on input prompt."""
    return f"[LLM Response Generated for Prompt Length: {len(prompt_text)} characters]"

# ==========================================
# 1. Zero-Shot Prompt Template
# ==========================================
zero_shot_template = """Answer the question based only on the following context. If the answer cannot be determined from the context, state "I cannot answer this based on the provided text."

Context:
{context}

Question: {question}

Answer:"""

zero_shot_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=zero_shot_template
)

# ==========================================
# 2. Few-Shot Prompt Template
# ==========================================
few_shot_template = """Answer the question based on the context. Follow the style and format shown in the examples.

Example 1:
Context: The company reported a revenue increase of 15% in Q3 2023, driven by cloud service growth.
Question: What drove the revenue increase in Q3 2023?
Answer: Revenue grew by 15% primarily due to expansion in cloud services.

Example 2:
Context: The new policy requires all remote workers to log into the VPN before accessing internal servers.
Question: What must remote workers do to access internal servers?
Answer: They are required to authenticate via the corporate VPN.

Context:
{context}

Question: {question}

Answer:"""

few_shot_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=few_shot_template
)

# ==========================================
# 3. Role-Based Prompt Template
# ==========================================
role_based_template = """You are a Senior Technical Document Analyst with expertise in synthesizing complex information with high technical precision. 

Your Task:
1. Thoroughly analyze the provided context.
2. Formulate a precise, concise, and direct answer to the user's question.
3. Strictly base your answer on the context provided; do not assume or extrapolate information.

Context:
{context}

Technical Query: {question}

Analyst Insights:"""

role_based_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=role_based_template
)

# ==========================================
# Testing & Comparison Execution
# ==========================================
if __name__ == "__main__":
    sample_context = "All employees working remotely must register their devices with IT before December 1st."
    
    # Assignment Requirement: Test using 5 questions
    test_questions = [
        "What must remote employees do before December 1st?",
        "Who needs to register their devices?",
        "What happens if an employee registers after December 1st?",
        "Which department handles device registration?",
        "Can employees use personal unregistered devices for work?"
    ]

    print("==================================================")
    print("      PART 1: PROMPT ENGINEERING TESTING         ")
    print("==================================================\n")

    for i, q in enumerate(test_questions, 1):
        print(f"--- Question {i}: {q} ---")
        
        # Format prompts
        p1 = zero_shot_prompt.format(context=sample_context, question=q)
        p2 = few_shot_prompt.format(context=sample_context, question=q)
        p3 = role_based_prompt.format(context=sample_context, question=q)
        
        print("\n[Zero-Shot Prompt Created]")
        print("[Few-Shot Prompt Created]")
        print("[Role-Based Prompt Created]\n")
        print("-" * 50)

    print("\nPrompt templates generated successfully!")