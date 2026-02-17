# CE49X: Introduction to Computational Thinking and Data Science for Civil Engineers
## Lab Assignment 2: Analyzing Career Pathways of Bogazici CE Graduates
### Spring 2026 | Dr. Eyuphan Koc | Bogazici University

---

**Due Date:** One week from the lab session
**Total Points:** 100
**Submission:** Upload your Jupyter notebook (.ipynb) to Moodle. All code cells must be executed with visible output.

---

## Background

Bogazici University's Department of Civil Engineering has produced generations of graduates who have pursued remarkably diverse career paths. In this lab, you will use Python's data science ecosystem — **pandas**, **NumPy**, **matplotlib**, and **string processing** — to clean, analyze, and visualize career trajectory data from our alumni.

You will work with two datasets provided as Python dictionaries (no external files needed). Simply copy them into your notebook to get started.

---

## Dataset 1: Alumni Career Records

```python
import pandas as pd
import numpy as np

alumni_records = [
    {"id": "CE-2020-001", "name": "  Ayse YILMAZ  ", "gender": "F", "sector": "Structural Engineering", "location": "Istanbul", "salary_tl": 42000, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-002", "name": "mehmet kaya", "gender": "M", "sector": "Geotechnical Engineering", "location": "Ankara", "salary_tl": 38500, "grad_year": 2020, "satisfaction": "Neutral"},
    {"id": "CE-2020-003", "name": "ELIF   DEMIR", "gender": "F", "sector": "Graduate Studies (Domestic)", "location": "Istanbul", "salary_tl": None, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-004", "name": "can oezturk", "gender": "M", "sector": "Construction Management", "location": "Izmir", "salary_tl": 35000, "grad_year": 2020, "satisfaction": "Unsatisfied"},
    {"id": "CE-2020-005", "name": "  ZEYNEP arslan ", "gender": "F", "sector": "Graduate Studies (Abroad)", "location": "Boston, USA", "salary_tl": None, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-006", "name": "burak   celik", "gender": "M", "sector": "Structural Engineering", "location": "Istanbul", "salary_tl": 44500, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-007", "name": "SELIN SAHIN  ", "gender": "F", "sector": "Transportation Planning", "location": "Ankara", "salary_tl": 39000, "grad_year": 2020, "satisfaction": "Neutral"},
    {"id": "CE-2020-008", "name": "emre yildiz", "gender": "M", "sector": "Finance / Consulting", "location": "Istanbul", "salary_tl": 55000, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-009", "name": "deniz  KARA", "gender": "F", "sector": "Environmental Engineering", "location": "Bursa", "salary_tl": 36000, "grad_year": 2020, "satisfaction": "Neutral"},
    {"id": "CE-2020-010", "name": "  ali guenes  ", "gender": "M", "sector": "Entrepreneurship", "location": "Istanbul", "salary_tl": 28000, "grad_year": 2020, "satisfaction": "Unsatisfied"},
    {"id": "CE-2020-011", "name": "basak AYDIN", "gender": "F", "sector": "Structural Engineering", "location": "Istanbul", "salary_tl": 41000, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-012", "name": "OGUZ   polat ", "gender": "M", "sector": "Geotechnical Engineering", "location": "Trabzon", "salary_tl": 34000, "grad_year": 2020, "satisfaction": "Neutral"},
    {"id": "CE-2020-013", "name": "irem erdogan", "gender": "F", "sector": "Graduate Studies (Abroad)", "location": "London, UK", "salary_tl": None, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-014", "name": " MURAT  aksoy", "gender": "M", "sector": "Construction Management", "location": "Antalya", "salary_tl": 37500, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-015", "name": "ceren   BAYRAK ", "gender": "F", "sector": "Graduate Studies (Domestic)", "location": "Istanbul", "salary_tl": None, "grad_year": 2020, "satisfaction": "Neutral"},
    {"id": "CE-2020-016", "name": "hakan  KORKMAZ", "gender": "M", "sector": "Transportation Planning", "location": "Istanbul", "salary_tl": 40000, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-017", "name": "GAMZE   tekin", "gender": "F", "sector": "Structural Engineering", "location": "Ankara", "salary_tl": 43500, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-018", "name": "tolga YALCIN  ", "gender": "M", "sector": "Entrepreneurship", "location": "Istanbul", "salary_tl": 62000, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-019", "name": "  naz dogan", "gender": "F", "sector": "Graduate Studies (Abroad)", "location": "Munich, Germany", "salary_tl": None, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-020", "name": "SERKAN   ozkan", "gender": "M", "sector": "Finance / Consulting", "location": "Istanbul", "salary_tl": 58000, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-021", "name": "yagmur  KURT ", "gender": "F", "sector": "Environmental Engineering", "location": "Mersin", "salary_tl": 33000, "grad_year": 2020, "satisfaction": "Unsatisfied"},
    {"id": "CE-2020-022", "name": "cem   ACAR", "gender": "M", "sector": "Structural Engineering", "location": "Kocaeli", "salary_tl": 39500, "grad_year": 2020, "satisfaction": "Neutral"},
    {"id": "CE-2020-023", "name": "  PINAR  tas  ", "gender": "F", "sector": "Graduate Studies (Domestic)", "location": "Ankara", "salary_tl": None, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-024", "name": "onur KILIC", "gender": "M", "sector": "Construction Management", "location": "Istanbul", "salary_tl": 41000, "grad_year": 2020, "satisfaction": "Satisfied"},
    {"id": "CE-2020-025", "name": "  ece   YILDIRIM", "gender": "F", "sector": "Structural Engineering", "location": "Istanbul", "salary_tl": 45000, "grad_year": 2020, "satisfaction": "Satisfied"},
]

df = pd.DataFrame(alumni_records)
```

---

## Dataset 2: Messy Survey Responses (Raw Text)

```python
survey_responses = """
CE-2020-001 | structural eng. | 5yrs exp | salary:42000TL | PE_license:yes
CE-2020-002 | geotech eng | 4.5 yrs exp | salary:38500 TL | PE_license:no
CE-2020-006 | structural eng. | 5 yrs exp | salary:44500TL | PE_license:YES
CE-2020-008 | finance/consulting | 3yrs exp | salary:55000 TL | PE_license:N/A
CE-2020-011 | structural eng | 4 yrs exp | salary:41000TL | PE_license:yes
CE-2020-014 | construction mgmt | 5yrs exp | salary:37500 TL| PE_license:no
CE-2020-017 | structural eng. | 5 yrs exp | salary:43500TL | PE_license:Yes
CE-2020-018 | entrepreneurship | 3.5yrs exp | salary:62000 TL | PE_license:no
CE-2020-020 | finance/consulting | 2 yrs exp | salary:58000TL | PE_license:N/A
CE-2020-024 | construction mgmt | 4.5 yrs exp | salary:41000 TL | PE_license:no
"""
```

---

## Part A: Data Cleaning with String Methods (25 points)

### Task 1: Name Standardization (10 points)

The `name` column is messy — inconsistent capitalization, extra whitespace, and irregular spacing. Write a function that cleans each name into **proper title case** with single spaces.

```
"  Ayse YILMAZ  "    →  "Ayse Yilmaz"
"mehmet kaya"         →  "Mehmet Kaya"
"ELIF   DEMIR"        →  "Elif Demir"
"can oezturk"         →  "Can Oezturk"
```

**(a)** Write a function `clean_name(name)` that:
1. Strips leading/trailing whitespace
2. Collapses multiple internal spaces into a single space
3. Converts to title case

Apply it to the entire `name` column using `.apply()` and display the first 10 rows showing the before and after.

**(b)** Using string methods on the `id` column, extract just the **numeric ID** (the last 3 digits) into a new column called `id_num`. For example, `"CE-2020-001"` should produce the integer `1`.

**(c)** Create a new column `initials` containing each person's initials (e.g., `"Ayse Yilmaz"` → `"A.Y."`). Use the cleaned name column and string operations.

---

### Task 2: Parsing Messy Survey Data with String Operations and Regex (15 points)

The `survey_responses` string contains semi-structured text data that needs to be parsed into a clean DataFrame.

**(a)** Split the raw text into individual records (one per line, skipping empty lines). For each record, use `.split('|')` to separate the fields. Create a DataFrame called `survey_df` with columns: `id`, `sector`, `years_exp`, `salary`, `pe_license`.

**(b)** Clean each column:
- `id`: Strip whitespace
- `sector`: Strip whitespace and convert to lowercase
- `years_exp`: Extract the numeric value (e.g., `"5yrs exp"` → `5.0`, `"4.5 yrs exp"` → `4.5`). Use `re.findall()` or `re.search()` with an appropriate pattern.
- `salary`: Extract the numeric salary value (e.g., `"salary:42000TL"` → `42000`). Use a regex pattern.
- `pe_license`: Standardize to boolean (`True`/`False`). Treat `"yes"`, `"Yes"`, `"YES"` as `True`; `"no"` as `False`; `"N/A"` as `None` (use `np.nan`).

**(c)** Display the cleaned `survey_df` and print a summary: how many alumni hold a PE license, what is the average years of experience, and what is the mean salary among survey respondents.

---

## Part B: Data Analysis with Pandas and NumPy (30 points)

### Task 3: Career Sector Analysis (10 points)

Using the original `df` DataFrame:

**(a)** Use `.value_counts()` to count the number of alumni in each career sector. What is the most common career path? What percentage of graduates remained in traditional engineering roles (Structural, Geotechnical, Environmental, Transportation, Construction Management)?

**(b)** Group the data by `gender` and `sector` using `.groupby()`. Create a **cross-tabulation** (using `pd.crosstab()`) showing the count of male and female graduates in each sector. Which sector has the highest proportion of female graduates?

**(c)** Compute the mean salary by sector (excluding `None` values for graduate students). Which sector pays the most on average? Use `.groupby()` and `.mean()`.

---

### Task 4: Location and Salary Analysis (10 points)

**(a)** Using string methods on the `location` column, create a new boolean column `is_abroad` that is `True` if the location contains a comma (indicating "City, Country" format). How many graduates are working/studying abroad?

**(b)** For graduates with salary data (not `None`), compute using NumPy:
- Mean salary (`np.mean`)
- Median salary (`np.median`)
- Standard deviation (`np.std`)
- The salary range (max − min)

Print the results formatted with f-strings to zero decimal places and include "TL" as the unit.

**(c)** Create a new column `salary_category` that classifies each employed graduate as:
- `"Below Average"` if salary < mean salary
- `"Above Average"` if salary >= mean salary
- `"N/A"` if salary is `None`

How many graduates fall into each category?

---

### Task 5: Satisfaction Analysis (10 points)

**(a)** Compute the overall satisfaction distribution (count and percentage for each level: Satisfied, Neutral, Unsatisfied).

**(b)** Group by `sector` and compute the satisfaction rate (proportion of "Satisfied" responses) for each sector. Which sector has the highest satisfaction rate? Which has the lowest?

**(c)** Among graduates who left traditional engineering (Finance/Consulting and Entrepreneurship), what is the satisfaction rate? Compare it to the satisfaction rate of those in traditional engineering roles. Write a one-sentence interpretation of the comparison.

---

## Part C: Data Visualization with Matplotlib (30 points)

For all plots, follow these conventions:
- Use the OO interface (`fig, ax = plt.subplots()`)
- Include `ax.grid(True, alpha=0.3)`
- Add descriptive titles, axis labels, and legends where appropriate
- Use `plt.tight_layout()` before `plt.show()`

---

### Task 6: Career Distribution Visualization (10 points)

**(a)** Create a **horizontal bar chart** showing the number of alumni in each career sector, sorted from most to least common. Use `color='steelblue'` for the bars. Add the count as a text annotation at the end of each bar.

**(b)** Create a **pie chart** showing the proportion of graduates in three broad categories:
- **Traditional Engineering** (Structural, Geotechnical, Environmental, Transportation, Construction Management)
- **Graduate Studies** (Domestic + Abroad)
- **Non-Engineering** (Finance/Consulting + Entrepreneurship)

Use `colors=['steelblue', 'indianred', 'goldenrod']` and display percentages on each slice with `autopct='%1.1f%%'`.

---

### Task 7: Salary and Gender Analysis (10 points)

**(a)** Create a **box plot** comparing salary distributions across the career sectors that have salary data. Rotate x-axis labels 45 degrees for readability. Title: "Salary Distribution by Career Sector (Class of 2020)".

**(b)** Create a **grouped bar chart** showing the count of male vs. female graduates in each of the three broad categories defined in Task 6(b). Use `steelblue` for male and `indianred` for female. Include a legend.

---

### Task 8: Multi-Panel Summary Dashboard (10 points)

Create a single figure with a **2x2 grid of subplots** (`fig, axes = plt.subplots(2, 2, figsize=(14, 10))`) containing:

1. **Top-left**: Bar chart of satisfaction levels (Satisfied, Neutral, Unsatisfied) with counts.
2. **Top-right**: Scatter plot of years of experience vs. salary (from `survey_df`), colored by PE license status.
3. **Bottom-left**: Histogram of salary distribution for all employed graduates (use 8 bins, `edgecolor='black'`).
4. **Bottom-right**: Bar chart showing average salary by broad career category.

Add a main title with `fig.suptitle("Bogazici CE Class of 2020 — Career Pathways Dashboard", fontsize=14, fontweight='bold')`.

---

## Part D: Reflection (15 points)

### Task 9: Written Analysis (15 points)

Write a short report (200–300 words) addressing the following, based on the results from your analysis:

**(a)** What are the three most significant findings from the career data? Reference specific numbers or visualizations from your analysis.

**(b)** Suppose the department wants to use this data to advise incoming students about career options. What are two limitations of drawing conclusions from this dataset? Think about sample size, data quality, and representativeness.

**(c)** Propose one additional data field that would make this analysis more useful (e.g., GPA, internship experience, specialization track). Explain what question it would help answer and which Python tool (pandas groupby, matplotlib visualization, regex parsing, etc.) you would use to analyze it.

---

## Submission Checklist

- [ ] All code cells executed with visible output
- [ ] Names properly cleaned and standardized (Task 1)
- [ ] Survey data parsed using regex and string methods (Task 2)
- [ ] Pandas analysis complete with groupby, crosstab, and value_counts (Tasks 3–5)
- [ ] All 6 plots rendered with proper formatting (Tasks 6–8)
- [ ] Written reflection addresses all three parts (Task 9)
- [ ] Notebook is well-organized with markdown headers for each part

---

### Questions?
**Dr. Eyuphan Koc**
eyuphan.koc@bogazici.edu.tr
