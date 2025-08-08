eva_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "evaluate_response",
            "schema": {
                "type": "object",
                "properties": {
                    "dimensions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "评分": {
                                    "type": "number",
                                    "description": "from 0 to 10"
                                },
                                "分析": {
                                    "type": "string",
                                    "description": "Brief summary referencing toolcall results"
                                },
                                "提升建议": {
                                    "type": "string",
                                    "description": "Provide actionable suggestions if rating is below 8; otherwise state 'Not required.'"
                                }
                            },
                            "required": ["评分", "分析", "提升建议"],
                            "additionalProperties": False
                        }

                    },
                    "summary": {"type": "string"}
                },
                "required": ["dimensions", "summary"],
                "additionalProperties": False
            },
            "strict": True
        }
    }


gen_format = [
    {
        "type": "json_schema",
        "json_schema": {
            "name": "generate_response",
            "schema": {
                "type": "object",
                "properties": {
                    "目标": {
                        "type": "string",
                        "description": "Clearly define the problem to be solved or the goal to be achieved with the use of AI"
                    },
                },
                "required": ["目标"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "json_schema",
        "json_schema": {
            "name": "generate_response",
            "schema": {
                "type": "object",
                "properties": {
                    "核心方案": {
                        "type": "string",
                        "description": "Provide a succinct overview of the key strategies or solutions driven by AI"
                    },
                },
                "required": ["核心方案"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "json_schema",
        "json_schema": {
            "name": "generate_response",
            "schema": {
                "type": "object",
                "properties": {
                    "执行步骤": {
                        "type": "string",
                        "description": "Outline 5-8 main steps necessary to implement the proposal, including timelines or priorities"
                    },
                },
                "required": ["执行步骤"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "json_schema",
        "json_schema": {
            "name": "generate_response",
            "schema": {
                "type": "object",
                "properties": {
                    "资源需求": {
                        "type": "string",
                        "description": "Identify the resources required, including personnel, technology, and financial support"
                    },
                },
                "required": ["资源需求"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "json_schema",
        "json_schema": {
            "name": "generate_response",
            "schema": {
                "type": "object",
                "properties": {
                    "风险与应对": {
                        "type": "string",
                        "description": "Recognize 2-3 major risks and suggest mitigation approaches"
                    },
                },
                "required": ["风险与应对"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "json_schema",
        "json_schema": {
            "name": "generate_response",
            "schema": {
                "type": "object",
                "properties": {
                    "预期效果": {
                        "type": "string",
                        "description": "Briefly describe the anticipated outcomes of the proposal, with quantitative measures whenever possible"
                    }
                },
                "required": ["预期效果"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]























                    



