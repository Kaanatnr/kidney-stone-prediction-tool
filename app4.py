import math
import pandas as pd
import streamlit as st
from model_utils import (
    load_model,
    predict,
    selected_feature_names,
    selected_medians,
    selected_imputed_values,
)

st.set_page_config(
    page_title="Kidney Stone Prediction Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Human-readable feature dictionary ----------
FEATURE_META = {
    "RIDEXMON": ("Examination season", "Demographic / timing", 0.006),
    "RIAGENDR": ("Sex", "Demographic", 0.031),
    "RIDAGEYR": ("Age", "Demographic", 0.081),
    "RIDRETH1": ("Race / ethnicity", "Demographic", 0.039),
    "DMDHRGND": ("Household reference person's sex", "Household", 0.007),
    "DMDHHSZB": ("Children aged 6–17 in household", "Household", 0.004),
    "DMDHRBR4": ("Household reference person's country of birth", "Household", 0.005),
    "BMIWT": ("Body-weight measurement note", "Measurement quality", 0.001),
    "BMXLEG": ("Upper leg length", "Body measurement", 0.019),
    "BMXWAIST": ("Waist circumference", "Body measurement", 0.042),
    "BMXSUB": ("Subscapular skinfold", "Body measurement", 0.005),
    "SMQ040": ("Current cigarette smoking", "Lifestyle", 0.007),
    "SMD100MN": ("Menthol cigarette indicator", "Lifestyle", 0.004),
    "SMD100NI": ("Cigarette nicotine content", "Lifestyle", 0.003),
    "DRQSDIET": ("Currently following a special diet", "Diet behavior", 0.004),
    "DRD350C": ("Crayfish eaten in last 30 days", "Seafood", 0.000),
    "DRD370AQ": ("Breaded fish frequency", "Seafood", 0.002),
    "DRD370E": ("Cod eaten in last 30 days", "Seafood", 0.001),
    "DRD370H": ("Mackerel eaten in last 30 days", "Seafood", 0.001),
    "DRD370LQ": ("Porgy frequency", "Seafood", 0.001),
    "DRD370OQ": ("Sea bass frequency", "Seafood", 0.002),
    "AVGTSUGR": ("Average daily total sugars", "Nutrition", 0.008),
    "AVGTATOC": ("Average daily vitamin E", "Nutrition", 0.007),
    "AVGTBCAR": ("Average daily beta-carotene", "Nutrition", 0.008),
    "AVGTVB2": ("Average daily riboflavin (vitamin B2)", "Nutrition", 0.003),
    "AVGTFA": ("Average daily folic acid", "Nutrition", 0.003),
    "AVGTVC": ("Average daily vitamin C", "Nutrition", 0.010),
    "AVGTCALC": ("Average daily calcium", "Nutrition", 0.006),
    "AVGTMAGN": ("Average daily magnesium", "Nutrition", 0.015),
    "AVGTSODI": ("Average daily sodium", "Nutrition", 0.004),
    "AVGTALCO": ("Average daily alcohol", "Lifestyle", 0.010),
    "AVGTS060": ("Average daily caproic acid (6:0)", "Nutrition", 0.008),
    "AVGTM181": ("Average daily oleic acid (18:1)", "Nutrition", 0.008),
    "AVGTM201": ("Average daily eicosenoic acid (20:1)", "Nutrition", 0.006),
    "AVGTWS": ("Main tap-water source code", "Diet / environment", 0.003),
}

SHAP_PATTERNS = [
    ("Age", "years", 41, "above", "more positive model contribution", "RIDAGEYR"),
    ("Waist circumference", "cm", 97, "above", "more positive model contribution", "BMXWAIST"),
    ("Total sugars", "g/day", 98.9, "above", "more positive model contribution", "AVGTSUGR"),
    ("Sodium", "mg/day", 3000, "above", "more positive model contribution", "AVGTSODI"),
    ("Magnesium", "mg/day", 272, "above", "more negative model contribution", "AVGTMAGN"),
    ("Calcium", "mg/day", 800, "above", "more negative model contribution", "AVGTCALC"),
    ("Vitamin C", "mg/day", 100, "above", "more negative model contribution", "AVGTVC"),
]

st.markdown("""
<style>
:root{--navy:#0c2d4d;--blue:#15658f;--teal:#158aa1;--ink:#172334;--muted:#667085;--line:#e4eaf0;--soft:#f5f8fb;--warn:#fff7df;--green:#176b55;--red:#a43d3d}
[data-testid="stAppViewContainer"]{background:#f6f8fb;color:var(--ink)}
[data-testid="stHeader"]{background:rgba(0,0,0,0)}
.block-container{max-width:1190px;padding-top:1.15rem;padding-bottom:4rem}
.hero{background:linear-gradient(128deg,#0d2948 0%,#155d82 68%,#188fa3 100%);border-radius:22px;padding:34px 38px 30px;color:white;margin-bottom:18px;box-shadow:0 16px 40px rgba(17,46,76,.16)}
.hero h1{margin:0 0 10px;font-size:42px;line-height:1.05;letter-spacing:-.025em}.hero p{margin:0;opacity:.92;max-width:900px;line-height:1.6;font-size:15px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:22px}.metric{padding:14px 16px;border:1px solid rgba(255,255,255,.2);border-radius:14px;background:rgba(255,255,255,.08)}.metric b{display:block;font-size:22px}.metric span{font-size:11px;opacity:.78;text-transform:uppercase;letter-spacing:.06em}
.badge{display:inline-block;font-size:10px;font-weight:800;letter-spacing:.08em;padding:6px 9px;border-radius:999px;background:#fff2bf;color:#755500;margin-bottom:13px}
.info-card{background:white;border:1px solid var(--line);border-radius:17px;padding:18px 20px;box-shadow:0 8px 24px rgba(22,45,72,.045);margin:10px 0}.info-card h4{margin:0 0 5px;color:var(--navy)}.small-note{font-size:12px;color:var(--muted);line-height:1.55}
.result-card{background:white;border:1px solid var(--line);border-radius:20px;padding:26px;box-shadow:0 12px 32px rgba(22,45,72,.08)}.result-value{font-size:64px;font-weight:850;line-height:1;color:var(--navy);letter-spacing:-.04em}.result-label{color:var(--muted);font-weight:650;margin-top:6px}
.scorebar{height:12px;background:#e9eef3;border-radius:999px;overflow:hidden;margin:18px 0 8px}.scorefill{height:100%;background:linear-gradient(90deg,#1b8ca1,#156b96);border-radius:999px}
.warning{background:var(--warn);border:1px solid #ecd996;color:#69521b;border-radius:13px;padding:13px 14px;font-size:12px;line-height:1.55}.okbox{background:#edf8f4;border:1px solid #cce9de;color:#285e50;border-radius:13px;padding:13px 14px;font-size:12px;line-height:1.55}
.process{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:14px 0 18px}.process-step{background:#fff;border:1px solid var(--line);border-radius:14px;padding:13px;min-height:105px}.process-step b{display:block;color:var(--navy);margin-bottom:5px}.process-step span{font-size:11px;color:var(--muted);line-height:1.4}
footer{visibility:hidden} div[data-testid="stButton"] button{width:100%;border-radius:12px;min-height:48px;font-weight:800;background:linear-gradient(135deg,#145d86,#188da2);color:white;border:0}
@media(max-width:800px){.hero h1{font-size:32px}.metrics{grid-template-columns:repeat(2,1fr)}.hero{padding:26px 22px}.result-value{font-size:52px}.process{grid-template-columns:1fr}}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* =========================================================
   FINAL VISUAL OVERRIDES — consistent light page + blue controls
   ========================================================= */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"]{
  background:#f6f8fb !important;
  color:#102a43 !important;
  color-scheme:light !important;
}

/* Hero */
.hero, .hero *{color:#ffffff !important;}
.hero p{color:#ffffff !important;opacity:1 !important;}
.hero .badge{color:#755500 !important;}

/* Section headings and ordinary text */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6{
  color:#0c2d4d !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li{
  color:#23384d !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p{
  color:#6f7d8d !important;
}

/* ALL widget labels must remain readable on the light page */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] > label,
label, label p{
  color:#0c2d4d !important;
  -webkit-text-fill-color:#0c2d4d !important;
  font-weight:650 !important;
}

/* Number inputs — same blue family as dropdowns */
[data-testid="stNumberInput"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] button{
  background:#145d86 !important;
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
  border-color:#145d86 !important;
  box-shadow:none !important;
}
[data-testid="stNumberInput"] > div > div{
  border:1px solid #145d86 !important;
  border-radius:11px !important;
  overflow:hidden !important;
}
[data-testid="stNumberInput"] input{caret-color:#ffffff !important;}
[data-testid="stNumberInput"] button svg{
  fill:#ffffff !important;
  color:#ffffff !important;
}
[data-testid="stNumberInput"] > div > div:focus-within{
  box-shadow:0 0 0 2px rgba(21,138,161,.28) !important;
  border-color:#188da2 !important;
}

/* Select boxes — matching blue with white text */
[data-testid="stSelectbox"] [data-baseweb="select"] > div{
  background:#145d86 !important;
  color:#ffffff !important;
  border:1px solid #145d86 !important;
  border-radius:11px !important;
  box-shadow:none !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover{
  background:#176f97 !important;
  border-color:#176f97 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] div{
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] svg{
  fill:#ffffff !important;
  color:#ffffff !important;
}

/* Opened select menus */
ul[role="listbox"], div[role="listbox"],
li[role="option"], li[role="option"] *{
  background:#145d86 !important;
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
}
li[role="option"]:hover, li[role="option"]:hover *{
  background:#188da2 !important;
  color:#ffffff !important;
}

/* Radio labels */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p{
  color:#23384d !important;
  -webkit-text-fill-color:#23384d !important;
}

/* Help/question icons: filled blue circle with white icon */
[data-testid="stTooltipIcon"] button,
[data-testid="stTooltipHoverTarget"] button,
button[aria-label="Help"]{
  width:20px !important;
  height:20px !important;
  min-width:20px !important;
  min-height:20px !important;
  padding:0 !important;
  background:#15658f !important;
  border:1px solid #15658f !important;
  border-radius:50% !important;
  box-shadow:none !important;
}
[data-testid="stTooltipIcon"] svg,
[data-testid="stTooltipHoverTarget"] svg,
button[aria-label="Help"] svg{
  width:13px !important;
  height:13px !important;
  color:#ffffff !important;
  fill:#ffffff !important;
  stroke:#ffffff !important;
}

/* Tooltip popup: blue background, white copy */
[role="tooltip"], [role="tooltip"] *,
div[data-baseweb="popover"] [data-testid="stTooltipContent"],
div[data-baseweb="popover"] [data-testid="stTooltipContent"] *{
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
}
[role="tooltip"],
div[data-baseweb="popover"] [data-testid="stTooltipContent"]{
  background:#145d86 !important;
  border:1px solid #1b789e !important;
  border-radius:10px !important;
}

/* Expanders: clean white cards, dark title */
[data-testid="stExpander"]{
  background:#ffffff !important;
  border:1px solid #d8e4ec !important;
  border-radius:12px !important;
  overflow:hidden !important;
  box-shadow:none !important;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary{
  background:#ffffff !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary svg{
  color:#0c2d4d !important;
  fill:#0c2d4d !important;
  -webkit-text-fill-color:#0c2d4d !important;
  font-weight:700 !important;
}
[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] summary:focus-visible{
  outline:none !important;
  box-shadow:none !important;
}

/* Submit and other buttons */
[data-testid="stFormSubmitButton"] button,
[data-testid="stButton"] button{
  width:100% !important;
  min-height:50px !important;
  border-radius:12px !important;
  border:0 !important;
  background:linear-gradient(135deg,#145d86,#188da2) !important;
  color:#ffffff !important;
  font-weight:800 !important;
}
[data-testid="stFormSubmitButton"] button *,
[data-testid="stButton"] button *{
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
}

/* Tabs */
button[data-baseweb="tab"], button[data-baseweb="tab"] p{
  color:#425466 !important;
}
button[data-baseweb="tab"][aria-selected="true"],
button[data-baseweb="tab"][aria-selected="true"] p{
  color:#0c2d4d !important;
  font-weight:800 !important;
}

/* Credits */
.team-card{
  background:#ffffff;
  border:1px solid #d8e4ec;
  border-radius:16px;
  padding:20px 22px;
  margin-top:18px;
}
.team-card h3{margin-top:0;color:#0c2d4d !important;}
.team-card p{color:#425466 !important;line-height:1.65;}
.team-names{font-size:14px;line-height:1.75;color:#23384d !important;}

@media(max-width:800px){
  [data-testid="stNumberInput"] > div > div,
  [data-testid="stSelectbox"] [data-baseweb="select"] > div{min-height:46px !important;}
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model()

model = get_model()
selected = selected_feature_names(model)
medians = selected_medians(model)

st.markdown("""
<div class="hero">
  <div class="badge">RESEARCH USE ONLY</div>
  <h1>Kidney Stone Prediction Tool</h1>
  <p>Interactive companion to an explainable NHANES stacking-ensemble study. The app uses the supplied fitted model directly and translates technical NHANES variables into plain-language inputs.</p>
  <div class="metrics">
    <div class="metric"><b>33,327</b><span>Participants</span></div>
    <div class="metric"><b>3,128</b><span>Stone history</span></div>
    <div class="metric"><b>35</b><span>Selected features</span></div>
    <div class="metric"><b>0.723</b><span>Stacking ROC-AUC</span></div>
  </div>
  <div style="margin-top:16px;font-size:12px;opacity:.88"><b>Study &amp; Web Tool Team:</b> Rifat Burak Ergül, Kaan Atanur, Bora Atıcı, Sıla Tekerek, Aybüke Canöz, M. Fırat Özervarlı and collaborators.</div>
</div>
""", unsafe_allow_html=True)

prediction_tab, how_tab, guide_tab, study_tab = st.tabs(["Prediction", "How it works", "Feature guide", "Study & model"])

with prediction_tab:
    st.markdown("### Enter participant information")
    st.caption("The most useful inputs are shown first. Survey-specific variables are optional; when unknown, the fitted pipeline uses its training-set median exactly as it did during model development.")

    with st.form("prediction_form"):
        st.markdown("#### 1) About the participant")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age (years)", 20, 80, 49, 1, help="Age at examination.")
        with c2:
            sex_label = st.selectbox("Sex", ["Female", "Male"], index=0)
            sex = {"Male": 1, "Female": 2}[sex_label]
        with c3:
            eth_label = st.selectbox("Race / ethnicity", [
                "Non-Hispanic White", "Non-Hispanic Black", "Mexican American", "Other Hispanic", "Other / multiracial"
            ], index=0)
            eth = {"Mexican American":1,"Other Hispanic":2,"Non-Hispanic White":3,"Non-Hispanic Black":4,"Other / multiracial":5}[eth_label]
        with c4:
            exam_label = st.selectbox("Time of year examined", ["May–October", "November–April"], index=0, help="NHANES grouped examinations into two six-month periods.")
            exam = 2 if exam_label == "May–October" else 1

        st.markdown("#### 2) Body measurements")
        b1, b2, b3 = st.columns(3)
        with b1:
            waist = st.number_input("Waist circumference (cm)", 40.0, 220.0, 98.0, 0.5, help="Measure around the abdomen with a flexible tape. The model uses the NHANES waist measure.")
        with b2:
            leg = st.number_input("Upper leg length (cm)", 20.0, 65.0, 38.5, 0.1, help="Approximate length of the upper leg. Leave the study-median default if this is unavailable.")
        with b3:
            sub = st.number_input("Skinfold below the shoulder blade (mm)", 0.0, 80.0, 21.1, 0.1, help="Subscapular skinfold thickness. This usually requires calipers; leave the study-median default if unknown.")

        st.markdown("#### 3) Lifestyle and nutrition")
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            smoke_label = st.selectbox("Do you currently smoke cigarettes?", ["Not at all", "Some days", "Every day"], index=0)
            smoke = {"Every day":1,"Some days":2,"Not at all":3}[smoke_label]
        with n2:
            special_label = st.selectbox("Are you following a special diet?", ["No", "Yes", "Don't know"], index=0, help="For example a weight-loss, low-sodium, low-fat, diabetic, vegetarian/vegan or other structured diet.")
            special = {"Yes":1,"No":2,"Don't know":9}[special_label]
        with n3:
            sugar = st.number_input("Average total sugars (g/day)", 0.0, 600.0, 94.8, 1.0)
        with n4:
            sodium = st.number_input("Average sodium (mg/day)", 0.0, 15000.0, 3112.0, 10.0)

        n5, n6, n7, n8 = st.columns(4)
        with n5:
            magnesium = st.number_input("Average magnesium (mg/day)", 0.0, 2000.0, 268.0, 1.0)
        with n6:
            calcium = st.number_input("Average calcium (mg/day)", 0.0, 5000.0, 816.0, 5.0)
        with n7:
            vitc = st.number_input("Average vitamin C (mg/day)", 0.0, 3000.0, 62.25, 1.0)
        with n8:
            vite = st.number_input("Average vitamin E (mg/day)", 0.0, 100.0, 6.8, 0.1)

        with st.expander("Alcohol calculator (optional feature engineering)", expanded=False):
            st.caption("The model requires grams of pure alcohol per day. You can enter that directly or let the app estimate it from drinks per week.")
            alcohol_mode = st.radio("Alcohol input method", ["Enter grams/day directly", "Estimate from drinks/week"], horizontal=True)
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                alcohol_direct = st.number_input("Alcohol (g/day)", 0.0, 300.0, 0.0, 0.5)
            with ac2:
                beer_week = st.number_input("330 mL beers / week", 0.0, 50.0, 0.0, 1.0)
            with ac3:
                wine_week = st.number_input("150 mL wine glasses / week", 0.0, 50.0, 0.0, 1.0)
            with ac4:
                spirits_week = st.number_input("45 mL spirit servings / week", 0.0, 50.0, 0.0, 1.0)
            st.caption("Estimator assumptions: beer 5% ABV, wine 12% ABV, spirits 40% ABV; ethanol density 0.789 g/mL. This is a transparent arithmetic conversion, not an AI estimate.")

        with st.expander("More nutrition inputs", expanded=False):
            x1, x2, x3 = st.columns(3)
            with x1:
                bcar = st.number_input("Beta-carotene (mcg/day)", 0.0, 50000.0, 1044.25, 10.0)
                b2 = st.number_input("Riboflavin / vitamin B2 (mg/day)", 0.0, 20.0, 1.817, 0.1)
            with x2:
                folic = st.number_input("Folic acid (mcg/day)", 0.0, 3000.0, 138.5, 5.0)
                sfa6 = st.number_input("Caproic acid / saturated fat 6:0 (g/day)", 0.0, 10.0, 0.22, 0.01, help="A small saturated fatty acid found mainly in dairy fats. Leave the study-median default if unknown.")
            with x3:
                mfa181 = st.number_input("Oleic acid / monounsaturated fat 18:1 (g/day)", 0.0, 200.0, 23.099, 0.1, help="A common monounsaturated fat found in foods such as olive oil.")
                mfa201 = st.number_input("Eicosenoic acid / monounsaturated fat 20:1 (g/day)", 0.0, 20.0, 0.231, 0.01, help="A less common monounsaturated fatty acid. Leave the study-median default if unknown.")

        with st.expander("Optional household and survey details", expanded=False):
            st.caption("These fields were selected by the fitted model but are less practical in routine use. Choosing 'Unknown' lets the model's own median imputer handle them.")
            h1, h2, h3 = st.columns(3)
            with h1:
                hhsex_label = st.selectbox("Sex of the household reference person", ["Unknown", "Male", "Female"], index=0, help="In NHANES this is the adult designated as the household reference person.")
                hhsex = {"Unknown":None,"Male":1,"Female":2}[hhsex_label]
                children_known = st.selectbox("Number of children aged 6–17 in the household", ["Unknown", "0", "1", "2", "3", "4", "5+"])
                children = {"Unknown":None,"0":0,"1":1,"2":2,"3":3,"4":4,"5+":5}[children_known]
            with h2:
                birth_label = st.selectbox("Household reference person born in", ["Unknown", "United States", "Outside the United States"], index=0)
                birth = {"Unknown":None,"United States":1,"Outside the United States":2}[birth_label]
                weight_note_label = st.selectbox("Any special note during body-weight measurement?", ["Unknown / not reported", "Could not obtain", "Scale capacity exceeded", "Clothing affected measurement", "Medical device / appliance affected measurement"], index=0)
                weight_note = {"Unknown / not reported":None,"Could not obtain":1,"Scale capacity exceeded":2,"Clothing affected measurement":3,"Medical device / appliance affected measurement":4}[weight_note_label]
            with h3:
                water_label = st.selectbox("Main source when drinking tap water", ["Community / city supply", "Well or rain cistern", "Spring", "Do not drink tap water", "Other", "Don't know"], index=0)
                water = {"Community / city supply":1,"Well or rain cistern":2,"Spring":3,"Do not drink tap water":4,"Other":91,"Don't know":99}[water_label]

        with st.expander("Optional smoking-product details", expanded=False):
            s1, s2 = st.columns(2)
            with s1:
                menthol_label = st.selectbox("If you smoke: menthol or non-menthol cigarettes?", ["Unknown / not applicable", "Non-menthol", "Menthol"], index=0)
                menthol = {"Unknown / not applicable":None,"Non-menthol":0,"Menthol":1}[menthol_label]
            with s2:
                nicotine = st.number_input("If known: cigarette nicotine content (mg)", 0.0, 3.0, 1.0, 0.1, help="If you do not know this, leave the default. For non-smokers the app marks this field missing and uses the fitted imputer.")

        with st.expander("Optional seafood questions", expanded=False):
            st.caption("NHANES asked about consumption in the previous 30 days. These variables have relatively low global SHAP importance in the fitted model.")
            f1, f2, f3 = st.columns(3)
            with f1:
                cray_label = st.selectbox("Crayfish in the last 30 days", ["Unknown", "No", "Yes"], index=0)
                crayfish = {"Unknown":None,"No":2,"Yes":1}[cray_label]
                breaded = st.number_input("Breaded fish products: times in last 30 days", 0, 30, 2, 1, help="If unknown, leaving the study-median default of 2 preserves the fitted-model convention.")
            with f2:
                cod_label = st.selectbox("Cod in the last 30 days", ["Unknown", "No", "Yes"], index=0)
                cod = {"Unknown":None,"No":2,"Yes":1}[cod_label]
                mack_label = st.selectbox("Mackerel in the last 30 days", ["Unknown", "No", "Yes"], index=0)
                mackerel = {"Unknown":None,"No":2,"Yes":1}[mack_label]
            with f3:
                porgy = st.number_input("Porgy / sea-bream-type fish: times in last 30 days", 0, 30, 2, 1)
                seabass = st.number_input("Sea bass: times in last 30 days", 0, 30, 1, 1)

        submitted = st.form_submit_button("Predict with the fitted stacking model")

    if submitted:
        if alcohol_mode == "Estimate from drinks/week":
            beer_g = 330 * 0.05 * 0.789
            wine_g = 150 * 0.12 * 0.789
            spirits_g = 45 * 0.40 * 0.789
            alcohol = ((beer_week * beer_g) + (wine_week * wine_g) + (spirits_week * spirits_g)) / 7.0
            alcohol_source = "Estimated from drinks/week"
        else:
            alcohol = alcohol_direct
            alcohol_source = "Entered directly"

        # For non-smokers, cigarette-product variables are treated as unavailable and left to fitted imputation.
        if smoke_label == "Not at all":
            menthol_value = None
            nicotine_value = None
        else:
            menthol_value = menthol
            nicotine_value = nicotine

        values = {
            "RIDEXMON": exam,
            "RIAGENDR": sex,
            "RIDAGEYR": age,
            "RIDRETH1": eth,
            "DMDHRGND": hhsex,
            "DMDHHSZB": children,
            "DMDHRBR4": birth,
            "BMIWT": weight_note,
            "BMXLEG": leg,
            "BMXWAIST": waist,
            "BMXSUB": sub,
            "SMQ040": smoke,
            "SMD100MN": menthol_value,
            "SMD100NI": nicotine_value,
            "DRQSDIET": special,
            "DRD350C": crayfish,
            "DRD370AQ": breaded,
            "DRD370E": cod,
            "DRD370H": mackerel,
            "DRD370LQ": porgy,
            "DRD370OQ": seabass,
            "AVGTSUGR": sugar,
            "AVGTATOC": vite,
            "AVGTBCAR": bcar,
            "AVGTVB2": b2,
            "AVGTFA": folic,
            "AVGTVC": vitc,
            "AVGTCALC": calcium,
            "AVGTMAGN": magnesium,
            "AVGTSODI": sodium,
            "AVGTALCO": alcohol,
            "AVGTS060": sfa6,
            "AVGTM181": mfa181,
            "AVGTM201": mfa201,
            "AVGTWS": water,
        }

        probability, pred_class, base = predict(model, values)
        pct = probability * 100
        label = "Kidney stone history present" if pred_class == 1 else "Kidney stone history absent"
        imputed = selected_imputed_values(model, values)
        supplied = sum(1 for f in selected if values.get(f) is not None)
        imputed_count = len(selected) - supplied

        # Simple transparent derived summaries. These are DISPLAY-ONLY and not fed to the fitted classifier.
        sodium_magnesium = sodium / magnesium if magnesium > 0 else math.nan
        calcium_magnesium = calcium / magnesium if magnesium > 0 else math.nan
        waist_leg = waist / leg if leg > 0 else math.nan

        st.markdown("---")
        st.subheader("Prediction result")
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown(f"""
            <div class="result-card">
              <div class="result-value">{pct:.1f}%</div>
              <div class="result-label">Fitted stacking-model probability output</div>
              <div class="scorebar"><div class="scorefill" style="width:{max(1,min(100,pct)):.1f}%"></div></div>
              <div style="font-weight:800;color:#0e2d4f;margin-top:12px">Model classification: {label}</div>
            </div>
            """, unsafe_allow_html=True)
        with right:
            st.markdown("#### Base-learner outputs")
            cols = st.columns(3)
            display_names = {"lr":"Logistic", "rf":"Random Forest", "xgb":"XGBoost"}
            for col, (name, value) in zip(cols, base.items()):
                col.metric(display_names.get(name, name), f"{value*100:.1f}%")
            st.markdown(f"""
            <div class="okbox"><b>Input handling:</b> {supplied}/35 selected features were explicitly supplied; {imputed_count}/35 were handled by the model's fitted median imputer. Alcohol input: {alcohol_source}.</div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="warning"><b>Important:</b> This percentage is the fitted classifier's probability output, not a clinically validated absolute risk estimate. The model was internally evaluated and has not been externally validated. Do not use it for diagnosis or treatment decisions.</div>
            """, unsafe_allow_html=True)

        st.markdown("#### Derived summaries")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Alcohol used by model", f"{alcohol:.1f} g/day")
        d2.metric("Sodium : magnesium", f"{sodium_magnesium:.1f}" if math.isfinite(sodium_magnesium) else "—")
        d3.metric("Calcium : magnesium", f"{calcium_magnesium:.2f}" if math.isfinite(calcium_magnesium) else "—")
        d4.metric("Waist : upper-leg ratio", f"{waist_leg:.2f}" if math.isfinite(waist_leg) else "—")
        st.caption("The three ratios are simple engineered summaries shown for interpretation only. They are NOT passed into the fitted classifier, because adding new predictors after training would change the published model.")

        st.markdown("#### Manuscript SHAP-pattern context")
        st.caption("These are reported model-derived inflection regions, not clinical cutoffs or treatment targets.")
        pattern_rows = []
        value_lookup = {
            "RIDAGEYR": age, "BMXWAIST": waist, "AVGTSUGR": sugar, "AVGTSODI": sodium,
            "AVGTMAGN": magnesium, "AVGTCALC": calcium, "AVGTVC": vitc,
        }
        for name, unit, threshold, relation, direction, code in SHAP_PATTERNS:
            val = value_lookup[code]
            reached = val >= threshold
            pattern_rows.append({
                "Feature": name,
                "Entered value": f"{val:.1f} {unit}",
                "Reported region": f"~{threshold:g} {unit} and above",
                "Pattern in manuscript": direction if reached else "Entered value is below the reported inflection region",
            })
        st.dataframe(pd.DataFrame(pattern_rows), use_container_width=True, hide_index=True)

        with st.expander("See the 35 model values after missing-value handling"):
            rows = []
            for code in selected:
                rows.append({
                    "Plain-language feature": FEATURE_META[code][0],
                    "Model code": code,
                    "Value used": round(float(imputed[code]), 4),
                    "Source": "Entered / derived" if values.get(code) is not None else "Fitted median imputation",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with how_tab:
    st.markdown("## How the app turns answers into a prediction")
    st.markdown("""
    <div class="process">
      <div class="process-step"><b>1. Plain-language inputs</b><span>The app converts ordinary answers such as sex, smoking status and food frequency into the numeric format used by NHANES.</span></div>
      <div class="process-step"><b>2. Simple feature engineering</b><span>Optional drinks/week are converted into grams of ethanol per day. A few ratios are calculated only for display and interpretation.</span></div>
      <div class="process-step"><b>3. Missing-value handling</b><span>If an optional model field is unknown, the exact fitted median imputer stored inside the PKL supplies the training-set median.</span></div>
      <div class="process-step"><b>4. Scaling + feature selection</b><span>The fitted StandardScaler standardizes values, then the stored L1-based selector retains the same 35 features used at training time.</span></div>
      <div class="process-step"><b>5. Stacking ensemble</b><span>Logistic Regression, Random Forest and XGBoost feed a Logistic Regression meta-learner that returns the final probability output.</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What feature engineering is done here?")
    st.markdown("""
- **Dietary averages:** the study model was developed using average daily nutrient values derived from two 24-hour recalls. The site therefore asks for *average daily* intake rather than a single meal or single-day value.
- **Alcohol conversion:** if the user chooses the drinks/week option, the app converts beverage volume × alcohol-by-volume × ethanol density into grams of ethanol, then divides weekly intake by 7. The resulting **g/day** value is what enters the model.
- **Categorical encoding:** plain-language answers such as *Male/Female*, *Yes/No*, smoking frequency and tap-water source are mapped back to their original NHANES numeric codes before prediction.
- **Context-aware missingness:** cigarette menthol and nicotine values are left missing for non-smokers; the model's own fitted median imputer then handles them consistently with its pipeline.
- **Display-only derived ratios:** sodium:magnesium, calcium:magnesium and waist:upper-leg ratios are shown to help users understand the entered profile. **They do not modify the prediction**, because the published model was not retrained with these new ratios.
    """)
    st.info("This separation is intentional: useful derived summaries can be shown without silently changing the mathematical model reported in the manuscript.")

    st.markdown("### Why are some unusual questions included?")
    st.write("The fitted L1 selector retained a few NHANES-specific variables such as household details, body-measurement comments, cigarette-product characteristics and seafood questions. They are kept in an optional section for reproducibility. If a user does not know them, the stored pipeline performs median imputation rather than forcing the user to guess.")

with guide_tab:
    st.markdown("## 35 model features in plain language")
    st.caption("Technical NHANES codes are kept only for reproducibility. The app itself presents human-readable labels.")
    guide_rows = []
    for code in selected:
        friendly, group, shap = FEATURE_META[code]
        guide_rows.append({
            "Feature": friendly,
            "Group": group,
            "NHANES / model code": code,
            "Global SHAP importance": shap,
            "Fitted median": round(float(medians[code]), 4),
        })
    guide_df = pd.DataFrame(guide_rows).sort_values("Global SHAP importance", ascending=False)
    st.dataframe(guide_df, use_container_width=True, hide_index=True)

    st.markdown("### Highest global SHAP importance")
    chart_df = guide_df.head(12).set_index("Feature")[["Global SHAP importance"]]
    st.bar_chart(chart_df)
    st.caption("Global SHAP importance describes average model influence across the study cohort. It is not causality and does not indicate the direction of effect by itself.")

with study_tab:
    st.markdown("## Study and fitted model")
    st.markdown("""
**Study:** *Explainable Machine Learning Reveals Nonlinear Dietary and Metabolic Patterns Associated with Kidney Stone History: A Population-Based NHANES Study*

- **Dataset:** NHANES 2007–2018, adults aged ≥20 years.
- **Final cohort:** 33,327 participants; 3,128 with self-reported kidney stone history and 30,199 without.
- **Dietary processing:** average of two 24-hour dietary recalls.
- **Preprocessing stored in the PKL:** median imputation → standard scaling → L1-based feature selection.
- **Selected predictors:** 35.
- **Base learners:** Logistic Regression, Random Forest and XGBoost.
- **Meta-learner:** Logistic Regression.
- **Reported 10-fold stratified cross-validation ROC-AUC:** 0.723 for the stacking ensemble.
- **Explainability in the manuscript:** model-agnostic permutation SHAP applied directly to the final stacking classifier.
    """)
    st.markdown("""
    <div class="warning"><b>Research-use limitation:</b> The study is cross-sectional, the outcome is self-reported kidney stone history, class-weighted training was used, and external validation has not yet been reported. The output should therefore be described as a <i>model probability output</i>, not as a validated clinical absolute-risk score.</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="team-card">
      <h3>Study &amp; Web Tool Team</h3>
      <p class="team-names">
        Rifat Burak Ergül, Kaan Atanur, Bora Atıcı, Sıla Tekerek, Aybüke Canöz, M. Fırat Özervarlı,
        Şeyda Gül Özcan, Stamatios Katsimperis, Theodoros Spinos, Ali Talyshinkii, Lazaros Tzelves,
        Mehmet Fatih Sahin, Angelis Peteinaris, Lukasz Nowak, Engin Denizhan Demirkiran,
        Patrick Juliebø-Jones, Arman Tsaturyan, Begoña Ballesta Martinez, Amelia Pietropaolo,
        Bhaskar Somani, and Tzevat Tefik.
      </p>
      <p>The web tool is an interactive implementation of the fitted model supplied for the study and is intended for research communication and reproducibility.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Reproducibility note")
    st.write("This web app loads the supplied serialized fitted pipeline directly. It does not reconstruct a new score from group averages or SHAP thresholds. This preserves the exact learned estimator and preprocessing sequence stored in the model file.")
