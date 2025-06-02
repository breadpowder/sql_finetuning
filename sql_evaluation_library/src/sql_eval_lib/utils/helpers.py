# sql_evaluation_library/src/sql_eval_lib/utils/helpers.py
import re

def split_sql_statements(sql_script: str) -> list[str]:
    """
    Splits a potentially multi-statement SQL script into individual statements.
    Handles simple cases; might need refinement for complex SQL with embedded semicolons in strings.
    Removes comments and filters out empty statements.
    """
    if not sql_script:
        return []

    # Remove C-style block comments /* ... */
    sql_script = re.sub(r"/\*.*?\*/", "", sql_script, flags=re.DOTALL)
    
    # Remove single-line comments -- ... and # ...
    sql_script = re.sub(r"--[^\r\n]*", "", sql_script)
    sql_script = re.sub(r"#[^\r\n]*", "", sql_script) # MySQL style comments, just in case

    # Split by semicolon
    statements = sql_script.split(';')
    
    # Clean up: strip whitespace from each statement and filter out any empty strings
    # that may result from splitting (e.g., if the script ends with a semicolon)
    cleaned_statements = [stmt.strip() for stmt in statements if stmt.strip()]
    
    return cleaned_statements

# Placeholder for other utility functions that might be added later,
# e.g., more sophisticated SQL parsing or comparison logic if needed.
```

This implementation of `split_sql_statements` first removes common SQL comment types (block and single-line) and then splits by semicolon. It finishes by stripping whitespace and removing any resulting empty statements. This should be reasonably robust for the `sql_context` strings typically found in datasets like `gretelai/synthetic_text_to_sql`.The `sql_evaluation_library/utils.py` file with `split_sql_statements` has been created.

Now, I will implement the `ExecutionEvaluationModule` in `sql_evaluation_library/exec_evaluator.py`.
This module will use the `split_sql_statements` utility and interact with the placeholder `LangfuseManager`.
