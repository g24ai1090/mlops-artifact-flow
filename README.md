# mlops-artifact-flow

Project Structure
.
|-- src/
| |-- train.py
| |-- inference.py
| |-- utils.py
|-- config/
| |-- config.json |-- tests/
| |-- test_train.py
|-- .github/
|
|
|
|
|-- requirements.txt
|-- workflows/ |-- train.yml
|-- test.yml
|-- inference.yml
|-- README.md
Branching Instructions
To maintain a clean and traceable development history, you must follow the prescribed branching order. This ensures that the codebase from previous phases is always available when beginning the next one.
2
ML Ops Assignment 2
  Branching Guideline
You should not merge your branches back into main after each phase. Instead, follow this linear branching approach:
1. Start with the main branch (contains only the README).
2. Create a new branch classification branch from main.
3. Complete Phase 1 in classification branch.
4. Checkout a new branch test branch from classification branch. 5. Complete Phase 2 in test branch.
6. Checkout a new branch inference branch from test branch.
7. Complete Phase 3 in inference branch.
At no point should you merge back into main. Each branch builds upon the previous one in a linear order.
Summary: main → classification branch → test branch → inference branch
This ensures all code and workflows from earlier phases are retained without conflicts, and CI
workflows remain functional across branches.
Phase 1: Training Pipeline
Branch name: classification branch Task 1.1 Implement src/train.py:
• Load the digits dataset.
• Read hyperparameters from config/config.json.
• Train a LogisticRegression model with config parameters.
• Save the model as model train.pkl using pickle/joblib.dump().
Task 1.2 Test the script locally. Ensure the model file is saved correctly. Task 1.3 Create .github/workflows/train.yml to:
• Checkout the code repository.
• Set up Python.
• Install all required dependencies using requirements.txt.
• Run train.py
• Upload model train.pkl as an artifact using actions/upload-artifact
Phase 2: Testing using Pytest Branch name: test branch
Task 2.1 Write Unit Tests for Training Pipeline. Create a test script named test train.py inside a new tests/ directory. Your tests must validate the following components of your code:
3

ML Ops Assignment 2
 (a) Configuration File Loading
• Test that the configuration file (config/config.json) loads successfully. • Check that all required hyperparameters exist in the configuration:
– C (float): inverse regularization strength
– solver (string): optimization algorithm (e.g., "lbfgs") – max iter (int): number of training iterations
• Check that the values have the correct data types. (b) Model Creation
• Call your training function (e.g., train model(X, y, config)).
• Verify that the function returns a LogisticRegression object.
• Optionally,confirmtheobjecthasbeenfitted(e.g.,bycheckingattributeslike.coef or .classes ).
(c) Model Accuracy
• Train the model on the digits dataset and evaluate it on the same data.
• Check that the accuracy is above some threshold to verify correctness of training logic.
Task 2.2 Test the script locally.
Task 2.3 Create .github/workflows/test.yml to:
• Checkout the code repository.
• Set up Python.
• Install all required dependencies using requirements.txt. • Run the full test suite using pytest.
Phase 3: Inference and Multi-Job Workflow
Branch name: inference branch Task 3.1 Implement src/inference.py:
• Create a new Python script named inference.py and place it inside the src/ directory.
• The script should load the trained model saved as model train.pkl, which was pro-
duced during the training phase.
• It must then use this model to generate predictions on the digit classification dataset
used earlier (i.e., the same dataset from sklearn.datasets.load digits). Task 3.2 Test the script locally.
Task 3.3 Create a new workflow file named inference.yml inside the .github/workflows/ directory. This workflow must consist of three separate jobs, each dependent on the previous:
4

ML Ops Assignment 2
 (a) Test Cases Job:
• Runs all test cases implemented in Phase 2.
• This job must run first and the rest of the pipeline must only proceed if the test job is successful.
(b) Train Job:
• Executes the training script that generates the trained model.
• This job should only run after the test cases job has passed.
• The resulting model must be saved and uploaded as an artifact so that it can be reused in the next step.
(c) Inference Job:
• Downloads the model artifact created in the train job.
• Executes the inference script that generates predictions.
• This job must only run after the train job has successfully completed.
* For the above Phase make sure to use ’needs’ parameter for the jobs requiring previous execution.
