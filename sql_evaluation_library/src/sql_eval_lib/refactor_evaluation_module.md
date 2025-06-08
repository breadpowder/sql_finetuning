Refactor code under evaluation/ You must take consideration abstraction for llm evaluation and concrete implementation for this SQL evaluation project.

- Refactor classes llm_evaluator and exec_evaluator for abstract evaluation interface.
    -- for both files, You must not have any concrete SQL related evaluation and only common logics for llm-as-judge and execution-as-judge.

    -- Create concreate sql_llm_evaluator and sql_exec_evaluator under /sql subfolder which implement the abstract interface with sql evaluation logic.

- Similarly for /metrics, refactor base_metrics.py, execution_metrics.py. llm_metrics.py. Which must be abstract metrics for all llm evalution and MUST NOT related to sql. While sql_metrics.py provive additional sql related metrics including semantic and syntax metrics. Remove perfomancemetrics since it is not well defined.

- Refactor all tests/ following structure in the @python.mdc rules

- Remove not used python files and tests/ files.

