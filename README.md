# Sweat the Details: Exercise Breaks and AI Assistance in Code Summarization

## Purpose:

This is a replication package for our submission. We provide the stimuli (code files and AI summaries), participant data, qualitative analysis data (summary ratings), data management scripts, and data analysis scripts.


## Stimuli

- Stimuli are stored in `stimuli`
  - Code files
    - Code files with no defect seeded (`stimuli/clean/`)
      - `stimuli/clean/fast-excel` contains code files from the fast-excel github project
      - `stimuli/clean/spring-ai-alibaba` contains code files from the spring-ai-alibaba github project
    - Code files with a seeded defect (`stimuli/defect-seeded/`)
      - `stimuli/clean/fast-excel` contains code files from the fast-excel github project
      - `stimuli/clean/spring-ai-alibaba` contains code files from the spring-ai-alibaba github project
  - ChatGPT responses (`stimuli/chatgpt_responses_html/`)
    - contains responses from ChatGPT-4 that are displayed in our experiment
    - In our experiment, reponse from ChatGPT varied in bug detection correctness (see section 2.3 in our paper)
    - Each html filename contains a tag describing the bug detection correctness
      - "tp" means true positive
      - "tn" means true negative
      - "fp" means false positive
      - "fn" means false negative
    - Each number corresponds to a code file (see `stimuli/stimulus_to_num.csv`)
  - `stimuli/stimulus_to_num.csv`
    - contains the mapping of stim_src_id to Java Method/file stimuli used in the experiment


## Data Management - Python

- Data (`data_management/data`)
  - qualitative-scoring (`data_management/data/qualitative-scoring`)
    - qualitative analysis for summary quality
      - one file for each rater
  - `transformed_dataframe.csv`
    - contains the raw output from the experiment with particaipants filtered out (see paper for more info)
      - each row represents an observation
      - this was used for the mixed effectes analysis
  - `unfiltered_transformed_dataframe.csv`
    - contains the raw output from the experiment with no participants filtered out
- Stimuli Misc Info (`data_management/stimuli_misc_info`)
  - Gives info for each stimuli about
    - LOC for target method
    - LOC for whole file
    - Halstead complexity
    - Cyclomatic complexity
- T-tests and summary of data (`data_management/t-tests`)


## Preprocessed Data

- `preprocessed-data/full-data-version-2/programming-full-data.csv`
  - output from Data Management - Python (see that section above)
  - input for Data Analysis - R (see that section below)


## Data Analysis - R

- R scripts
  - List of scripts
    - `accuracy-bug-analysis`
      - bug detection accuary
    - `accuracy-summary-analysis`
      - summary quality
        - accuracy
        - completeness
        - conciseness
        - readability
    - `bug-accuracy-ai-correctness-analysis`
      - bug detection accuracy for the AI condition only
    - `RT-analysis`
      - response time for good quality summaries
  - To view our scripts already run with the result, use the html files
  - In order to run the R scripts, the `candidate-models/regression-models-programming.zip` file must be unzipped


## Candidate Models

- All candidate mixed effects models per dependent variable


## Quotes from participants

- `quotes/Summaries` from Participants - Mapping between P# and part_id.csv
  - maps quotes in the paper to participant ids


