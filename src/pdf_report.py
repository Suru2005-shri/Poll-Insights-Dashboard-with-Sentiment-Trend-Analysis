"""
pdf_report.py
-------------
Upgrade 2: Auto-generates a professional PDF report with:
  - Cover page
  - Key metrics summary
  - All 10 charts embedded
  - Statistical test results
  - Insights page

Install:  pip install fpdf2
Run:      python src/pdf_report.py
Output:   outputs/Poll_Results_Report.pdf
"""

import os
from datetime import datetime
from fpdf import FPDF
from src.analysis import (
    load_and_clean, overall_product_votes, nps_analysis,
    satisfaction_analysis, generate_insights
)
from src.stats_testing import run_full_significance_report


class PollReport(FPDF):
    """Custom PDF class with branded header/footer."""

    BRAND_BLUE  = (24,  95, 165)
    BRAND_DARK  = (30,  58, 95)
    BRAND_LIGHT = (230, 241, 251)
    GRAY        = (100, 100, 100)
    LINE_GRAY   = (220, 220, 220)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*self.BRAND_BLUE)
        self.rect(0, 0, 210, 10, "F")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 2)
        self.cell(0, 6, "Poll Results Visualizer - Confidential Report", ln=False)
        self.set_xy(0, 2)
        self.cell(200, 6, f"Page {self.page_no()}", align="R")
        self.ln(12)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*self.GRAY)
        self.cell(0, 6, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Poll Results Visualizer", align="C")

    def cover_page(self, n_respondents, n_regions, date_range):
        self.add_page()
        # Background header block
        self.set_fill_color(*self.BRAND_DARK)
        self.rect(0, 0, 210, 90, "F")

        self.set_font("Helvetica", "B", 28)
        self.set_text_color(255, 255, 255)
        self.set_xy(15, 25)
        self.cell(0, 12, "Poll Results Visualizer", ln=True)

        self.set_font("Helvetica", "", 14)
        self.set_text_color(*self.BRAND_LIGHT)
        self.set_x(15)
        self.cell(0, 8, "Product Preference Survey - Analytics Report", ln=True)

        # Stat boxes
        self.set_y(100)
        stats = [
            ("Respondents", str(n_respondents)),
            ("Regions",     str(n_regions)),
            ("Questions",   "5"),
            ("Charts",      "10"),
        ]
        for i, (label, val) in enumerate(stats):
            x = 15 + i * 47
            self.set_fill_color(*self.BRAND_LIGHT)
            self.rect(x, 100, 42, 28, "F")
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*self.BRAND_DARK)
            self.set_xy(x, 106)
            self.cell(42, 10, val, align="C")
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*self.GRAY)
            self.set_xy(x, 117)
            self.cell(42, 5, label, align="C")

        self.set_y(140)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.GRAY)
        self.cell(0, 6, f"Survey period: {date_range}", align="C", ln=True)
        self.cell(0, 6, f"Report generated: {datetime.now().strftime('%B %d, %Y')}", align="C")

    def section_title(self, title):
        self.ln(6)
        self.set_fill_color(*self.BRAND_BLUE)
        self.rect(10, self.get_y(), 3, 8, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*self.BRAND_DARK)
        self.set_x(16)
        self.cell(0, 8, title, ln=True)
        self.ln(2)

    def kv_row(self, key, value, highlight=False):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.GRAY)
        self.set_x(12)
        self.cell(70, 7, key)
        self.set_font("Helvetica", "B" if highlight else "", 10)
        self.set_text_color(*self.BRAND_DARK)
        self.cell(0, 7, str(value), ln=True)

    def insight_row(self, text):
        self.set_fill_color(*self.BRAND_LIGHT)
        self.set_x(12)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.BRAND_DARK)
        self.rect(12, self.get_y(), 186, 8, "F")
        self.set_x(15)
        self.cell(183, 8, text[:100], ln=True)
        self.ln(1)

    def embed_chart(self, img_path, caption, w=170):
        if not os.path.exists(img_path):
            return
        try:
            self.ln(3)
            x = (210 - w) / 2
            self.image(img_path, x=x, w=w)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(*self.GRAY)
            self.cell(0, 5, clean_text(caption), align="C", ln=True)

            self.ln(4)
        except Exception as e:
            print(f"Could not embed {img_path}: {e}")

    def stat_row(self, test_name, chi2_or_f, p_val, conclusion):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*self.BRAND_DARK)
        self.set_x(12)
        self.cell(0, 6, test_name, ln=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*self.GRAY)
        self.set_x(15)
        self.cell(0, 5, f"Statistic = {chi2_or_f}   |   p-value = {p_val}", ln=True)
        color = (15, 110, 86) if p_val < 0.05 else (180, 0, 0)
        self.set_text_color(*color)
        self.set_x(15)
        self.cell(0, 5, conclusion, ln=True)
        self.ln(2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(df, output_path="outputs/Poll_Results_Report.pdf"):
    os.makedirs("outputs", exist_ok=True)

    votes         = overall_product_votes(df)
    _, nps_score  = nps_analysis(df)
    sat           = satisfaction_analysis(df)
    insights      = generate_insights(df)
    stats_results = run_full_significance_report(df)

    winner     = votes.iloc[0]["product"]
    winner_pct = votes.iloc[0]["percentage"]
    date_range = f"{df['date'].min().strftime('%b %d')} – {df['date'].max().strftime('%b %d, %Y')}"

    pdf = PollReport()
  
        pdf.add_font("DejaVu", "", "C:/Users/SHRUTI/Documents/polll/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf", uni=True)

    pdf.set_font("DejaVu", "", 12)
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── COVER ─────────────────────────────────────────────────────────────────
    pdf.cover_page(len(df), df["region"].nunique(), date_range)

    # ── PAGE 2: EXECUTIVE SUMMARY ─────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Executive Summary")
    pdf.kv_row("Survey period:",       date_range)
    pdf.kv_row("Total respondents:",   f"{len(df):,}")
    pdf.kv_row("Regions covered:",     ", ".join(df["region"].unique()))
    pdf.kv_row("Age groups:",          ", ".join(sorted(df["age_group"].unique())))
    pdf.kv_row("Leading product:",     f"{winner} ({winner_pct}%)", highlight=True)
    pdf.kv_row("Net Promoter Score:",  f"{nps_score}", highlight=True)
    pdf.kv_row("Avg rating:",          f"{df['rating'].mean():.2f} / 10")

    pdf.ln(5)
    pdf.section_title("Key Insights")
    for ins in insights:
        # Strip emoji (fpdf latin-1 safe)
        clean = ins.encode("latin-1", errors="replace").decode("latin-1")
        pdf.insight_row(clean)

    # ── PAGES 3-4: CHARTS ─────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Vote Share Analysis")
    pdf.embed_chart("outputs/01_overall_votes.png", "Figure 1 — Overall product vote share")
    pdf.embed_chart("outputs/02_vote_pie.png",      "Figure 2 — Vote distribution (pie)")

    pdf.add_page()
    pdf.section_title("Regional & Demographic Analysis")
    pdf.embed_chart("outputs/03_region_stacked.png", "Figure 3 — Region-wise stacked preference")
    pdf.embed_chart("outputs/04_region_heatmap.png", "Figure 4 — Region × Product heatmap")

    pdf.add_page()
    pdf.section_title("Demographic Breakdown")
    pdf.embed_chart("outputs/05_age_grouped.png", "Figure 5 — Age group preference")

    pdf.add_page()
    pdf.section_title("Satisfaction & Loyalty")
    pdf.embed_chart("outputs/06_satisfaction.png", "Figure 6 — Customer satisfaction distribution")
    pdf.embed_chart("outputs/07_nps_donut.png",    "Figure 7 — Net Promoter Score breakdown")

    pdf.add_page()
    pdf.section_title("Trend & Rating Analysis")
    pdf.embed_chart("outputs/08_monthly_trend.png", "Figure 8 — Monthly voting trend")
    pdf.embed_chart("outputs/09_avg_rating.png",    "Figure 9 — Average rating by product")
    pdf.embed_chart("outputs/10_would_buy_again.png","Figure 10 — Would buy again")

    # ── STATISTICAL TESTS PAGE ────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Statistical Significance Tests")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(12)
    pdf.multi_cell(186, 5,
        "The following tests confirm whether observed differences in product preferences "
        "are statistically significant or could be due to chance. p < 0.05 is the standard "
        "threshold for claiming a real effect.")
    pdf.ln(4)

    r = stats_results["region_test"]
    pdf.stat_row(
        r["test"],
        f"chi2={r['chi2']}  dof={r['dof']}  Cramer's V={r['cramers_v']} ({r['effect']} effect)",
        r["p_value"],
        r["conclusion"]
    )
    a = stats_results["age_test"]
    pdf.stat_row(a["test"], f"chi2={a['chi2']}  dof={a['dof']}  Cramer's V={a['cramers_v']}", a["p_value"], a["conclusion"])

    g = stats_results["gender_test"]
    pdf.stat_row(g["test"], f"chi2={g['chi2']}  dof={g['dof']}  Cramer's V={g['cramers_v']}", g["p_value"], g["conclusion"])

    an = stats_results["anova"]
    pdf.stat_row(an["test"], f"F={an['f_statistic']}", an["p_value"], an["conclusion"])

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    pdf.output(output_path)
    print(f"PDF report saved → {output_path}")
    return output_path


if __name__ == "__main__":
    df = load_and_clean("data/poll_data.csv")
    generate_pdf_report(df)
