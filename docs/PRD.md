# Product Requirements Document (PRD)

## Problem Statement
In mass-casualty or high-volume emergency situations, medical staff must rapidly sort patients by treatment urgency using the START (Simple Triage and Rapid Treatment) protocol. Manual triage under time pressure is prone to error, and misclassifying a critical patient as stable can cost lives. You are tasked with building a machine learning model that automates this triage decision using patient vitals, consciousness scores, injury characteristics, and demographic data. This modeling approach must explicitly account for the unequal clinical costs of different classification errors using a specific cost matrix.

## Target Classes
The objective is to categorize emergency patients into the following four triage categories:
* **RED** — Immediate treatment required
* **YELLOW** — Delayed treatment may be possible
* **GREEN** — Minor injuries / ambulatory patients
* **BLACK** — Deceased or expectant patients

## Cost Matrix
| Actual \ Predicted    | RED   | YELLOW | GREEN | BLACK |
| :---                  | :---: | :---:  | :---: | :---: |
| **RED**               | 0     | 5      | 10    | 5     |
| **YELLOW**            | 2     | 0      | 3     | 2     |
| **GREEN**             | 1     | 1      | 0     | 1     |
| **BLACK**             | 2     | 2      | 2     | 0     |

## Required Evaluation Metrics
The following metrics must be reported:
* Accuracy
* Precision, Recall, F1-score - per class
* Macro-F1, Weighted-F1
* Confusion matrix
* Total misclassification cost

## Submission Format
* CSV File with required output
* A short README (max 1 page) explaining your approach and work done.
* Python Notebook with codes
* Can submit as a github repo/zip file.

### Filename Pattern
`{team_ID}_MM26ML03.csv`

### CSV Columns
The CSV file must contain the patient identifier and predicted triage category for each test patient, along with class probabilities. The submission must preserve the required patient ordering and contain all required prediction fields.

| Patient ID | Predicted Triage | RED Probability | YELLOW Probability | GREEN Probability | BLACK Probability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| P001 | RED | 0.87 | 0.08 | 0.03 | 0.02 |
