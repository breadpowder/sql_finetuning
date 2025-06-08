"""
Prompt management for evaluation strategies.

This module centralizes all prompts used in evaluation strategies,
making them easily configurable and extensible for different domains.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format the prompt template with the given arguments."""
        pass
    
    @abstractmethod
    def get_required_args(self) -> list[str]:
        """Return list of required arguments for this template."""
        pass


class StringPromptTemplate(PromptTemplate):
    """Simple string-based prompt template with placeholder substitution."""
    
    def __init__(self, name: str, template: str, description: str = ""):
        super().__init__(name, description)
        self.template = template
    
    def format(self, **kwargs) -> str:
        """Format the template using string format method."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required argument for prompt '{self.name}': {e}")
    
    def get_required_args(self) -> list[str]:
        """Extract placeholder names from the template string."""
        import re
        # Find all placeholders in format {arg_name}
        placeholders = re.findall(r'\{(\w+)\}', self.template)
        return list(set(placeholders))


class MultiPartPromptTemplate(PromptTemplate):
    """Template that combines multiple parts (e.g., system and user prompts)."""
    
    def __init__(self, name: str, parts: Dict[str, str], description: str = ""):
        super().__init__(name, description)
        self.parts = parts
    
    def format(self, **kwargs) -> Dict[str, str]:
        """Format all parts and return as dictionary."""
        formatted_parts = {}
        for part_name, template in self.parts.items():
            try:
                formatted_parts[part_name] = template.format(**kwargs)
            except KeyError as e:
                raise ValueError(f"Missing required argument for prompt '{self.name}', part '{part_name}': {e}")
        return formatted_parts
    
    def get_required_args(self) -> list[str]:
        """Extract placeholder names from all template parts."""
        import re
        all_placeholders = []
        for template in self.parts.values():
            placeholders = re.findall(r'\{(\w+)\}', template)
            all_placeholders.extend(placeholders)
        return list(set(all_placeholders))


# SQL Evaluation Prompts
SQL_LLM_EVALUATION_PROMPTS = {
    "sql_llm_evaluator_system": StringPromptTemplate(
        name="sql_llm_evaluator_system",
        description="System prompt for SQL LLM evaluation",
        template="""You are an expert SQL evaluator. Your task is to evaluate a generated SQL query based on several criteria.
Provide a score from 1 (worst) to 5 (best) for each criterion, along with detailed textual reasoning for each score.
Output your response strictly as a single JSON object. Do not include any text outside of this JSON object.

The JSON object should have the following structure, with scores as integers and reasoning as strings:
{{
    "semantic_correctness": {{ "score": <integer>, "reasoning": "<text>" }},
    "hallucinations_schema_adherence": {{ "score": <integer>, "reasoning": "<text>" }},
    "efficiency_conciseness": {{ "score": <integer>, "reasoning": "<text>" }},
    "overall_quality_readability": {{ "score": <integer>, "reasoning": "<text>" }}
}}"""
    ),
    
    "sql_llm_evaluator_user": StringPromptTemplate(
        name="sql_llm_evaluator_user",
        description="User prompt for SQL LLM evaluation",
        template="""Please evaluate the **Generated SQL Query** based on the provided context.

**1. Natural Language Question (SQL Prompt):**
{sql_prompt}

**2. Database Schema (SQL Context):**
```sql
{sql_context}
```

**3. Generated SQL Query (to be evaluated):**
```sql
{generated_sql}
```

**4. Ground Truth SQL Query (for reference, DO NOT SCORE THIS ONE, use it to inform your score of the Generated SQL):**
```sql
{ground_truth_sql}
```

Evaluate the **Generated SQL Query** against the criteria and provide your response in the specified JSON format.
Focus on:
- **Semantic Correctness:** Does the generated SQL accurately represent the intent of the Natural Language Question? Does it fetch the correct data as per the question and schema? Compare its logic with the Ground Truth SQL.
- **Hallucinations / Schema Adherence:** Does the generated SQL only use tables and columns defined in the Database Schema (SQL Context)? Are there any fabricated table or column names?
- **Efficiency and Conciseness:** Is the generated SQL efficient? Are there any redundant operations or overly complex structures compared to the Ground Truth SQL or optimal SQL practices? Is it concise?
- **Overall Quality & Readability:** Is the generated SQL well-formatted and easy to understand? Does it follow common SQL coding conventions? Considering all aspects, what is its overall quality?"""
    ),
    
    "sql_generation_system": StringPromptTemplate(
        name="sql_generation_system",
        description="System prompt for SQL generation",
        template="""You are an expert SQL writer. Given a database schema (CREATE TABLE statements) and a natural language question, your task is to write a syntactically correct SQL query that accurately answers the question based on the provided schema. Ensure the query is efficient and directly addresses the question. Only output the SQL query, with no additional explanation or markdown."""
    ),
    
    "sql_generation_user": StringPromptTemplate(
        name="sql_generation_user",
        description="User prompt for SQL generation",
        template="""Database Schema:
{sql_context}

Question:
{sql_prompt}"""
    )
}

# Generic evaluation prompts for other domains
GENERIC_EVALUATION_PROMPTS = {
    "generic_llm_evaluator_system": StringPromptTemplate(
        name="generic_llm_evaluator_system",
        description="Generic system prompt for LLM evaluation",
        template="""You are an expert evaluator for AI-generated content. Your task is to evaluate generated output based on several criteria relevant to the domain.
Provide scores and detailed reasoning for each criterion. Output your response as a JSON object with the following structure:

{{
    "accuracy": {{ "score": <number>, "reasoning": "<text>" }},
    "completeness": {{ "score": <number>, "reasoning": "<text>" }},
    "relevance": {{ "score": <number>, "reasoning": "<text>" }},
    "quality": {{ "score": <number>, "reasoning": "<text>" }}
}}

Score each criterion from 1 (worst) to 5 (best)."""
    ),
    
    "generic_llm_evaluator_user": StringPromptTemplate(
        name="generic_llm_evaluator_user",
        description="Generic user prompt for LLM evaluation",
        template="""Please evaluate the following generated output:

**Input Prompt:**
{prompt}

**Context:**
{context}

**Generated Output:**
{generated_output}

**Reference Output (if available):**
{reference_output}

Evaluate the generated output against the criteria and provide your response in the specified JSON format."""
    )
}


class PromptManager:
    """
    Manages prompt templates and provides easy access to formatted prompts.
    """
    
    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        # Load SQL evaluation prompts
        for name, template in SQL_LLM_EVALUATION_PROMPTS.items():
            self._templates[name] = template
        
        # Load generic evaluation prompts
        for name, template in GENERIC_EVALUATION_PROMPTS.items():
            self._templates[name] = template
    
    def register_template(self, template: PromptTemplate) -> None:
        """Register a new prompt template."""
        self._templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self._templates.get(name)
    
    def format_prompt(self, name: str, **kwargs) -> str:
        """Format a prompt template with the given arguments."""
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Prompt template '{name}' not found")
        return template.format(**kwargs)
    
    def list_templates(self) -> list[str]:
        """List all available template names."""
        return list(self._templates.keys())
    
    def get_required_args(self, name: str) -> list[str]:
        """Get required arguments for a template."""
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Prompt template '{name}' not found")
        return template.get_required_args()
    
    def create_multi_part_prompt(self, system_template: str, user_template: str, **kwargs) -> Dict[str, str]:
        """
        Create a multi-part prompt (e.g., system + user) by formatting separate templates.
        
        Args:
            system_template: Name of the system prompt template
            user_template: Name of the user prompt template
            **kwargs: Arguments for formatting both templates
            
        Returns:
            Dictionary with 'system' and 'user' keys containing formatted prompts
        """
        system_prompt = self.format_prompt(system_template, **kwargs)
        user_prompt = self.format_prompt(user_template, **kwargs)
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }


# Global prompt manager instance
prompt_manager = PromptManager() 