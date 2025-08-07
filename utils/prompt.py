eva_prompt = """Evaluate the provided report or content, rating each of the following six dimensions on a scale from **0 to 10**. For each dimension, provide: 

            * **A numerical rating** (0–10).
            * **Brief justification** based directly on the toolcall results.
            * **Suggestions for improvement** (if rating is below 8).

            Dimensions for evaluation:

            1. **语言正确性**:
            Assess accuracy in spelling, grammar, punctuation, and sentence structure based on the provided toolcall feedback.

            2. **逻辑结构**:
            Evaluate clarity, coherence, and logical flow of arguments as indicated by the toolcall insights.

            3. **信息价值**:
            Determine the relevance, originality, and usefulness of the content, guided by toolcall indicators.

            4. **可读性**:
            Rate how easy, engaging, and audience-appropriate the content is, utilizing toolcall readability scores.

            5. **合规安全**:
            Confirm the absence of harmful, offensive, or misleading content according to toolcall alerts and provide a safety rating.

            6. **目标契合度**:
            Judge alignment with audience expectations and needs as suggested by toolcall assessments.

            Make sure you provide your evaluations in the provided structured JSON format and response in Chinese.

            Ensure your assessment is comprehensive, balanced,  and clearly justified."""


gen_prompt = """Generate a detailed business proposal using AI technology for our clients. Make sure each section is clear and concise, while addressing all points specified.
            
            The proposal should address the following sections: 

            1. **目标**: Outline what problem this plan aims to solve or what objective it seeks to achieve.

            2. **核心方案**: Summarize the key approach or solution strategy that will be utilized.

            3. **执行步骤**: List 3-5 major steps to execute the plan, including timeframes or priorities.

            4. **资源需求**: Enumerate the people, technology, and financial support required to implement the plan.

            5. **风险与应对**: Identify 1-2 critical risks and propose mitigation strategies.

            6. **预期效果**: Briefly describe the expected outcomes of the plan, quantifying them if possible.

            Ensure each section is addressed thoroughly and is well-structured. 
            
            Tailor the content to fit the client’s industry and needs. Summarize the scope of the project and AI opportunities."""