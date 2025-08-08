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


gen_prompt = ["""
            Create a part of the comprehensive business proposal using AI for our clients, ensuring the content extends beyond 2000 words. Make sure each section is clear and concise, while addressing all points specified. The proposal should address the following sections: 
            1. **目标**: 
            Clearly articulate the specific problem your AI solution aims to resolve or the goal it seeks to achieve. Provide background information and context about why this problem or goal is significant for the client’s business and industry. Include examples and case studies that illustrate the impact of similar challenges.
            The proposal should be tailored to the client’s industry, highlighting relevant AI opportunities. Make sure you reply in Chinese as long as possible.
            """,
            """
            Create a part of the comprehensive business proposal using AI for our clients, ensuring the content extends beyond 2000 words. Make sure each section is clear and concise, while addressing all points specified. The proposal should address the following sections: 
            2. **核心方案**: 
            Offer an in-depth overview of the key AI strategies or solutions. Specify the particular technologies involved, such as natural language processing, machine learning algorithms, or neural networks. Discuss the rationale behind choosing these technologies, including their benefits and potential limitations. Provide detailed explanations of how these solutions will be integrated into the client’s current processes.
            The proposal should be tailored to the client’s industry, highlighting relevant AI opportunities. Make sure you reply in Chinese as long as possible.
            """,
            """
            Create a part of the comprehensive business proposal using AI for our clients, ensuring the content extends beyond 2000 words. Make sure each section is clear and concise, while addressing all points specified. The proposal should address the following sections: 
            3. **执行步骤**: 
            Lay out 5-8 comprehensive steps needed for the proposal’s execution. Clearly identify the technologies or tools required, such as TensorFlow, PyTorch, or specific AI APIs, and include timelines or priority levels for each step. Expand on each step by explaining what tasks need to be completed, who will be responsible, and how these steps align with the overall strategy.            
            The proposal should be tailored to the client’s industry, highlighting relevant AI opportunities. Make sure you reply in Chinese as long as possible.
            """,
            """
            Create a part of the comprehensive business proposal using AI for our clients, ensuring the content extends beyond 2000 words. Make sure each section is clear and concise, while addressing all points specified. The proposal should address the following sections: 
            4. **资源需求**: 
            List in detail the human resources (e.g., data scientists, AI engineers) and technical resources (e.g., GPUs, cloud platforms like AWS or Azure) needed, along with detailed budget estimates. Discuss how these resources will be procured, managed, and optimized to support the proposal goals effectively. Include potential partnerships or collaborations that may be beneficial.
            The proposal should be tailored to the client’s industry, highlighting relevant AI opportunities. Make sure you reply in Chinese as long as possible.
            """,
            """
            Create a part of the comprehensive business proposal using AI for our clients, ensuring the content extends beyond 2000 words. Make sure each section is clear and concise, while addressing all points specified. The proposal should address the following sections: 
            5. **风险与应对**: 
            Identify 2-3 significant risks, such as technology adoption challenges or data privacy issues, and provide detailed mitigation strategies. Expand on how these risks could affect the project and steps to be taken to minimize their impact. Include a risk matrix or analysis chart if applicable to visualize potential impacts and mitigation strategies.
            The proposal should be tailored to the client’s industry, highlighting relevant AI opportunities. Make sure you reply in Chinese as long as possible.
            """,
            """
            Create a part of the comprehensive business proposal using AI for our clients, ensuring the content extends beyond 2000 words. Make sure each section is clear and concise, while addressing all points specified. The proposal should address the following sections: 
            6. **预期效果**: 
            Qualitatively describe the anticipated impacts and benefits of the proposal, such as enhanced customer satisfaction, increased operational efficiency, or improved competitive advantage. Discuss how these outcomes will affect the client’s business and industry positively. Include anecdotal evidence or case studies showcasing similar successful implementations.
            The proposal should be tailored to the client’s industry, highlighting relevant AI opportunities. Make sure you reply in Chinese as long as possible.
            """]