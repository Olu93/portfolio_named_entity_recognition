# Task: Extract/NLP

**Est. time/effort:** 20 hours

## GitLab Repository

Please use the private repository to version control your code:

- **Repository:** https://git.toptal.com/screening-ops/Olusanmi-hundogan
- **Username:** Olusanmi-hundogan
- **Password:** b58816eb545ad6ec727f4ea5fe8ba476

## Task Description

**Expected delivery time:** 7 days

### Task Scope and Expectations

The purpose of this task is to analyze the data and build a model that solves the specified problem.

We want to see how you:
- Explore the data
- Identify the most important features
- Track experiments
- Summarize your findings
- Use the data to build a working AI solution

We want to see you understand how to:
- Set up a proper project infrastructure
- Show knowledge of simple problem solving
- Demonstrate knowledge of using your preferred frameworks and libraries

The task has certain objective target metrics, but your approach and presentation are also taken into account.

**Please make sure that your model is easy to run and reproduce.**

### Task Details

You are given a dataset containing text from news articles:

**Dataset:** https://drive.google.com/file/d/1--q8Land0cC3mMn0mqGV-wA_PAIBGrwO/view?usp=sharing

Each article has associated labels of **Persons**, **Organizations**, and **Locations**. Your task is to design a solution to extract these entities from the provided data.

#### Evaluation Criteria

- **Names and locations** should be evaluated only by matching multi-word entities
  - Example: "Joe Biden" is only true if your prediction is "Joe Biden", and not if your prediction is ["Joe", "Biden"]
- **Organizations** should be evaluated using both:
  - Multi-word prediction (i.e. "Department of Agriculture" is True if the true label is "Department of Agriculture")
  - Single-word comparison, i.e., ["Department", "of", "Agriculture"] compared to every single word in the true label

#### Requirements

1. **Model Performance:** You must show a considerable improvement over an out-of-the-box NLP model/solution
2. **Deployment:** You will need to make the final model and its pre-processing stages deployable as a REST API
3. **API Functionality:** Your REST API will be used to obtain predictions on the test set, so it should be functional for the interview

#### API Endpoint Specification

For test set testing, please create an endpoint that will:

- **Input:** Take as a parameter a CSV file that contains only "text" and "themes" columns
- **Processing:** Read the CSV into a dataframe and use the "text" column as the text for scoring
- **Output:** Produce a CSV file containing columns: `['persons', 'organizations', 'locations']` with your scored answers separated by semicolons
- **Requirements:** The resulting file must have the same number of rows as the input file, even if some of the answers are empty
- **Packaging:** Package the endpoint solution as a Docker container

#### Performance Testing

To test your deployed model for throughput and latency, you have a `locustfile.py`* performance test file which you will need to:

1. Execute on your running model
2. Report the results in the README file
3. Demonstrate how your solution can handle at least **20 concurrent users**

> *Note: This specific file was meant for `locust==2.13.0` which supports Python 3.7 and upward, any version of `pandas`. Here's the project's documentation page explaining how you should run it: https://docs.locust.io/en/stable/quickstart.html

## Milestones and Task Delivery

### Deadlines

- **Project submission deadline:** 7 days from the moment you receive the project requirements
- **Code submission:** Project code must be submitted within 7 days from the moment it was delivered to you by email
- **Pre-interview submission:** If you schedule your final interview after the 7-day deadline, make sure to submit your completed project and all code to the private repository before the deadline

### Important Notes

- **Late submissions:** Everything that is submitted after the deadline will not be taken into consideration
- **Review time:** To ensure sufficient review time, please commit all code or documents at least 12 hours before the meeting
- **Meeting attendance:** Please join the meeting room for this final interview on time. If you miss your interview without prior notice, your application may be closed