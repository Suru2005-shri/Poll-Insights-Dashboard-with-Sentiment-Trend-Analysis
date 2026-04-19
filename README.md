Poll Results Visualizer

> A complete, industry-oriented **Survey Data Analysis & Visualization** project built for placement portfolios.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-teal)
![Plotly](https://img.shields.io/badge/Plotly-5.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)

---

Problem Statement

Organizations collect thousands of poll/survey responses but struggle to:
- Identify **which product, candidate, or option leads**
- Understand **regional and demographic breakdowns**
- Track **satisfaction trends over time**
- Generate **actionable insights** quickly

##  Solution

A fully automated Python pipeline that:
1. Accepts poll data (CSV or synthetic)
2. Cleans and processes it
3. Performs multi-dimensional analysis
4. Generates 10 professional visualizations
5. Produces a live **Streamlit dashboard** with filters

---

##  Features

| Feature | Details |
|---|---|
|  Data Ingestion | CSV upload or synthetic data generation |
|  Data Cleaning | Deduplication, null handling, type parsing |
|  10 Chart Types | Bar, Pie, Stacked, Heatmap, Trend, Donut, Grouped |
|  Region Analysis | Preference breakdown by geographic region |
|  Demographics | Age group & gender analysis |
|  Satisfaction | 5-point scale + NPS score calculation |
|  Trend Analysis | Monthly voting trend per product |
|  Auto Insights | 8 key insights generated automatically |
|  Dashboard | Interactive Streamlit dashboard with live filters |

---

## 🛠️ Tech Stack

```
Python 3.9+
├── Data Layer:     Pandas, NumPy
├── Visualization:  Matplotlib, Seaborn, Plotly
├── Dashboard:      Streamlit
└── Output:         PNG charts + CSV exports
```

---

##  Folder Structure

```
Poll-Results-Visualizer/
│
├── data/
│   └── poll_data.csv          ← auto-generated or your CSV
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py       ← synthetic dataset creator
│   ├── analysis.py            ← all analysis functions
│   └── visualizer.py          ← chart generation (Matplotlib/Seaborn)
│
├── outputs/                   ← 10 saved PNG charts
│
├── notebooks/                 ← Jupyter EDA notebooks (optional)
│
├── app.py                     ← Streamlit dashboard (Plotly)
├── main.py                    ← CLI entry point
├── requirements.txt
└── README.md
```

---

##  Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/Poll-Results-Visualizer.git
cd Poll-Results-Visualizer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

##  How to Run

### Option A — CLI (generates all charts as PNG files)
```bash
python main.py
```
Charts saved to `outputs/`

### Option B — Interactive Dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

##  Charts Generated

| # | Chart | Purpose |
|---|---|---|
| 01 | Horizontal Bar | Overall vote share |
| 02 | Pie Chart | Vote distribution |
| 03 | Stacked Bar | Region-wise preference |
| 04 | Heatmap | Region × Product matrix |
| 05 | Grouped Bar | Age group preference |
| 06 | Bar Chart | Satisfaction distribution |
| 07 | Donut Chart | NPS score breakdown |
| 08 | Line Chart | Monthly trend |
| 09 | Bar Chart | Avg rating by product |
| 10 | Stacked Bar | Would buy again |

---

##  Sample Insights

-  **Product A leads** overall with 25.3% of all votes
-  Product A is most popular in the **North region**
-  **NPS Score: 17.3** — needs improvement
-  Average respondent rating: **5.52 / 10**

---

##  Real-World Use Cases

-  **Election Poll Analysis** — track candidate preferences
-  **Product Preference Surveys** — which SKU wins?
-  **Employee Satisfaction** — HR annual surveys
-  **Classroom Feedback** — post-course evaluation
-  **App Feature Voting** — what to build next



---

##  License

MIT License — free to use for learning and portfolio.
