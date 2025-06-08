Need dataset features.
Context and Goal:
Context
Datasets are used for offline evaluation
Dataset is a collection of DatasetItems
DatasetItem contains input, expected_output, and metadata
DatasetRun is an experiment run on a Dataset, it is identified by a unique name
DatasetRunItem links a DatasetItem to a Trace created during an experiment
Evaluation metrics of a DatasetRun are based on Scores associated with the Traces linked to run

Goal: Provide dataset mamagment capability (Create, Update, Delete)
a. Dataset can be created from hugging face or customer dataset in s3 by reading using pandas.  

b. Additional filtering and transformation using sql expression can be added to dataset processing.

c. Raw Source data can have different schema.

d. Concrete implementation can be for SQL evaluation dataset. (current reference
./sql_evaluation_library/scripts/prepare_dataset.py with a hugging face dataset)

e. When you design this, consider the future implementation which needs to be integrated with @evaluation.  Read doc (https://langfuse.com/docs/datasets/python-cookbook)

d. For testing, please finish unit test. then proceed with integration test, which bring up langfuse docker to create the sql dataset from hugging face using gretelai/synthetic_text_to_sql also test filter functionality.

Implementation:
Please add a folder dataset. 
Read example (https://langfuse.com/docs/datasets/get-started)

Some example code
langfuse.create_dataset(
    name="<dataset_name>",
    # optional description
    description="My first dataset",
    # optional metadata
    metadata={
        "author": "Alice",
        "date": "2022-01-01",
        "type": "benchmark"
    }
)

langfuse.create_dataset_item(
    dataset_name="<dataset_name>",
    input={ "text": "hello world" },
    expected_output={ "text": "hello world" },
    # link to a trace
    source_trace_id="<trace_id>",
    # optional: link to a specific span, event, or generation
    source_observation_id="<observation_id>"
)


Add folder and module accordingly for offline evaluation dataset.

