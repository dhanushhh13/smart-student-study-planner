"""Smart Student Study Planner - Streamlit app.

Run with:  streamlit run app.py
Needs: models/study_planner_model.joblib, models/feature_columns.joblib,
       data/student-mat.csv and src/recommend.py in the project folder.
"""
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
sys.path.append(str(BASE))
from src.recommend import suggest_improvements  # noqa: E402

# Typical error of the final model on the test set (marks). Update if it changes.
TYPICAL_ERROR = 2.2

st.set_page_config(page_title="Smart Student Study Planner", page_icon="📚", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(BASE / "models" / "study_planner_model.joblib")
    columns = joblib.load(BASE / "models" / "feature_columns.joblib")
    return model, columns


@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "data" / "student-mat.csv", sep=";")
    return df[df["G3"] > 0]  # same cleaning as the notebooks


def importance_by_input(model, columns):
    """Add up the forest's importance of one-hot columns back to the original input."""
    names = model.named_steps["prep"].get_feature_names_out()
    values = model.named_steps["model"].feature_importances_
    totals = {col: 0.0 for col in columns}
    for name, value in zip(names, values):
        short = name.split("__", 1)[1]
        for col in columns:
            if short == col or short.startswith(col + "_"):
                totals[col] += value
                break
    return pd.Series(totals).sort_values(ascending=False)


try:
    model, columns = load_model()
    data = load_data()
except FileNotFoundError as err:
    st.error(f"A required file is missing: {err.filename}. "
             "Check the models/ and data/ folders.")
    st.stop()

STUDY = {1: "Under 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "Over 10 hours"}
TRAVEL = {1: "Under 15 min", 2: "15 to 30 min", 3: "30 min to 1 hour", 4: "Over 1 hour"}
EDU = {0: "None", 1: "Primary school", 2: "Lower secondary",
       3: "Upper secondary", 4: "Higher education"}
YES_NO = ["no", "yes"]
JOBS = ["at_home", "health", "other", "services", "teacher"]

st.title("📚 Smart Student Study Planner")
st.write("Estimate a final math grade (0 to 20) from study habits and background, "
         "and see which habits are linked to a better result.")

tab_predict, tab_insights, tab_perf, tab_about = st.tabs(
    ["Predict", "Data Insights", "Model Performance", "About"])

# ------------------------------------------------------------------ Predict
with tab_predict:
    with st.form("student_form"):
        st.subheader("Your details")
        with st.expander("Study habits", expanded=True):
            c1, c2, c3 = st.columns(3)
            studytime = c1.select_slider("Weekly study time", options=[1, 2, 3, 4],
                                         value=2, format_func=STUDY.get)
            absences = c2.number_input("Absences this term", min_value=0, max_value=100,
                                       value=4, step=1)
            failures = c3.selectbox("Past class failures", [0, 1, 2, 3], index=0)
            c1, c2, c3, c4 = st.columns(4)
            higher = c1.selectbox("Wants higher education", YES_NO, index=1)
            paid = c2.selectbox("Extra paid classes", YES_NO, index=0)
            schoolsup = c3.selectbox("Extra school support", YES_NO, index=0)
            famsup = c4.selectbox("Family educational support", YES_NO, index=1)
        with st.expander("Lifestyle", expanded=True):
            c1, c2, c3 = st.columns(3)
            goout = c1.slider("Going out with friends (1 low, 5 high)", 1, 5, 3)
            freetime = c2.slider("Free time after school (1 low, 5 high)", 1, 5, 3)
            famrel = c3.slider("Family relationships (1 bad, 5 excellent)", 1, 5, 4)
            c1, c2, c3 = st.columns(3)
            walc = c1.slider("Weekend alcohol use (1 very low, 5 very high)", 1, 5, 2)
            dalc = c2.slider("Weekday alcohol use (1 very low, 5 very high)", 1, 5, 1)
            health = c3.slider("Health (1 very bad, 5 very good)", 1, 5, 4)
            c1, c2, c3 = st.columns(3)
            activities = c1.selectbox("Extracurricular activities", YES_NO, index=0)
            romantic = c2.selectbox("In a relationship", YES_NO, index=0)
            internet = c3.selectbox("Internet at home", YES_NO, index=1)
        with st.expander("Background"):
            c1, c2, c3, c4 = st.columns(4)
            age = c1.number_input("Age", min_value=15, max_value=22, value=17, step=1)
            sex = c2.selectbox("Sex", ["F", "M"])
            school = "GP"
            address = c4.selectbox("Home area", ["U", "R"],
                                   format_func={"U": "Urban", "R": "Rural"}.get)
            c1, c2, c3, c4 = st.columns(4)
            famsize = c1.selectbox("Family size", ["GT3", "LE3"],
                                   format_func={"GT3": "More than 3", "LE3": "3 or fewer"}.get)
            pstatus = c2.selectbox("Parents live", ["T", "A"],
                                   format_func={"T": "Together", "A": "Apart"}.get)
            guardian = c3.selectbox("Main guardian", ["mother", "father", "other"])
            reason = c4.selectbox("Reason for choosing school",
                                  ["course", "home", "other", "reputation"])
            c1, c2, c3, c4 = st.columns(4)
            medu = c1.select_slider("Mother's education", options=[0, 1, 2, 3, 4],
                                    value=3, format_func=EDU.get)
            fedu = c2.select_slider("Father's education", options=[0, 1, 2, 3, 4],
                                    value=2, format_func=EDU.get)
            mjob = c3.selectbox("Mother's job", JOBS, index=2)
            fjob = c4.selectbox("Father's job", JOBS, index=2)
            c1, c2, c3 = st.columns(3)
            traveltime = c1.select_slider("Home to school travel", options=[1, 2, 3, 4],
                                          value=1, format_func=TRAVEL.get)
            nursery = c2.selectbox("Attended nursery school", YES_NO, index=1)
        submitted = st.form_submit_button("Predict my grade", type="primary")

    if submitted:
        values = {
            "school": school, "sex": sex, "age": age, "address": address,
            "famsize": famsize, "Pstatus": pstatus, "Medu": medu, "Fedu": fedu,
            "Mjob": mjob, "Fjob": fjob, "reason": reason, "guardian": guardian,
            "traveltime": traveltime, "studytime": studytime, "failures": failures,
            "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher,
            "internet": internet, "romantic": romantic, "famrel": famrel,
            "freetime": freetime, "goout": goout, "Dalc": dalc, "Walc": walc,
            "health": health, "absences": int(absences),
        }
        try:
            student = pd.DataFrame([values])[columns]
            predicted = min(max(float(model.predict(student)[0]), 0.0), 20.0)
            suggestions = suggest_improvements(model, student)
        except Exception:
            st.error("Sorry, something went wrong while making the prediction. "
                     "Please check your inputs and try again.")
            st.stop()

        low, high = max(predicted - TYPICAL_ERROR, 0), min(predicted + TYPICAL_ERROR, 20)
        left, right = st.columns(2)
        left.metric("Predicted final grade", f"{predicted:.1f} / 20")
        right.write(f"The model is typically off by about {TYPICAL_ERROR} marks, "
                    f"so a likely range is **{low:.1f} to {high:.1f}**.")

        if predicted < 8 or predicted > 15:
            st.warning("Estimates are less reliable for very low or very high grades, "
                       "because the model tends to pull predictions toward the average.")
        if absences > 40:
            st.info("Absence counts above 40 are rare in the training data, "
                    "so this estimate is less reliable.")

        st.subheader("What could help")
        if suggestions:
            for s in suggestions:
                st.markdown(f"- **{s['action']}** ({s['from']} → {s['to']}): "
                            f"about **+{s['gain']:.1f} marks**")
        else:
            st.write("None of the habits we test would change the estimate noticeably. "
                     "Keep it up!")
        st.caption("Based on patterns in 357 students. These are associations, "
                   "not guarantees, and the effects are small.")

        with st.expander("Why this prediction?"):
            top = importance_by_input(model, columns).head(3).index.tolist()
            st.write(f"Across all students, the model relies most on: "
                     f"**{', '.join(top)}**. The suggestions above come from "
                     "re-running the model with one habit changed at a time.")

# ------------------------------------------------------------ Data Insights
with tab_insights:
    st.subheader("What the data shows")
    st.caption(f"{len(data)} students (38 students with a final grade of 0 were removed).")
    c1, c2 = st.columns(2)
    c1.write("Average final grade by past failures")
    c1.bar_chart(data.groupby("failures")["G3"].mean())
    c2.write("Average final grade by weekly study time level")
    c2.bar_chart(data.groupby("studytime")["G3"].mean())
    st.write("Distribution of final grades")
    st.bar_chart(data["G3"].value_counts().sort_index())
    st.write("Students with more past failures tend to score lower. "
             "Study time shows a smaller, less steady link.")

# --------------------------------------------------------- Model Performance
with tab_perf:
    st.subheader("How well does the model work?")
    st.write("Final model: tuned random forest. Scores on 72 held-out students "
             "(numbers from the modeling notebook):")
    st.table(pd.DataFrame(
        {"MAE (marks)": [2.32, 2.19], "RMSE (marks)": [2.77, 2.72], "R²": [0.20, 0.23]},
        index=["Linear regression", "Tuned random forest"]))
    st.write("Cross-validated results on the training data were similar (MAE about 2.4), "
             "so the small test set may flatter the model a little.")
    st.write("Where the model makes its biggest errors:")
    st.table(pd.DataFrame(
        {"Students": [12, 32, 25, 3], "Average miss (marks)": [3.85, 1.38, 2.11, 4.83],
         "Bias (marks)": [3.85, 1.12, -2.08, -4.83]},
        index=["Grades 5-8", "Grades 9-12", "Grades 13-16", "Grades 17-20"]))
    st.write("The model overestimates weak students and underestimates strong ones.")
    st.write("Which inputs matter most (random forest importance):")
    st.bar_chart(importance_by_input(model, columns).head(10))

# -------------------------------------------------------------------- About
with tab_about:
    st.subheader("About this project")
    st.write("This app predicts a student's final math grade (G3) from habits and "
             "background information, and shows which habits are linked to a better grade.")
    st.markdown(
        "- **Data:** UCI Student Performance (math course), 395 students; "
        "38 with a final grade of 0 (likely dropouts) were removed.\n"
        "- **Earlier grades (G1, G2) are not used**, because they would not be "
        "available early in the term and would leak the answer.\n"
        "- **Limits:** a small dataset from two Portuguese schools, mostly "
        "self-reported. Suggestions show associations, not causes.\n"
        "- **Not a verdict:** treat the prediction as a rough estimate, not a promise."
    )
