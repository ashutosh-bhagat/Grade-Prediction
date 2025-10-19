# Student Grade Prediction

This repository is a small Streamlit web app that predicts a student's final grade (G3) using a trained scikit-learn regression model.

Contents

- `app.py` — Streamlit app that collects input features from the user and predicts the final grade using saved artifacts (`scaler.pkl`, `student_grade_model.pkl`).
- `student-grade.csv` — Dataset (original student performance CSV) used for training (if you retrain locally).
- `student_grade_model.pkl` — Saved scikit-learn regression model (expected to be in the workspace).
- `scaler.pkl` — Saved scikit-learn StandardScaler used to scale input features before prediction.

Goal

- Provide a quick interactive UI to estimate a student's final grade (G3) from features like age, study time, previous grades, and binary attributes.

Running the app (Windows / PowerShell)

Make sure you have Conda and the appropriate environment installed (the repository was developed with Python 3.12 on Conda). If you use a different environment manager, adapt the commands.

From PowerShell in the project folder run:

```powershell
# Run using your conda python so the right packages are used
C:/Users/Ashu/anaconda3/Scripts/conda.exe run -p C:\Users\Ashu\anaconda3 --no-capture-output python -m streamlit run app.py
```

Then open your browser at:

- Local URL: http://localhost:8501

Notes

- If you see "The term 'streamlit' is not recognized", it means your shell PATH isn't pointing to the environment where Streamlit is installed. Using `python -m streamlit` inside the correct Conda environment (example above) avoids that issue.
- When running `app.py` directly with `python app.py` you will get warnings because Streamlit session state and the browser UI are intended to run under `streamlit run`.

Retraining the model (optional)

This repo may or may not contain a training script. To add the attendance percentage feature or to retrain the model yourself, follow these steps:

1. Create a training script (example outline):

   - Load `student-grade.csv` using pandas
   - Compute `attendance_percentage = max(0, (TOTAL_CLASSES - absences) / TOTAL_CLASSES * 100)`
   - Encode binary columns (`internet`, `romantic`) as 1/0 and `sex` to `sex_M` as (sex == 'M').astype(int)
   - Use features: `['age','studytime','internet','romantic','G1','G2','sex_M','attendance_percentage']`
   - Fit a `StandardScaler` on X and a `LinearRegression` (or other estimator) on scaled X
   - Save artifacts with `joblib.dump(scaler, 'scaler.pkl')` and `joblib.dump(model, 'student_grade_model.pkl')`

2. Run the training script inside the same Conda environment as Streamlit to ensure consistent package versions:

```powershell
C:/Users/Ashu/anaconda3/Scripts/conda.exe run -p C:\Users\Ashu\anaconda3 --no-capture-output python train_model.py
```

3. After training, start the app with the `streamlit run` command above. The app expects `scaler.pkl` and `student_grade_model.pkl` in the working directory.

Including attendance in the app

- The Streamlit UI can compute `attendance_percentage` from `absences`. If you retrain the model to include `attendance_percentage`, update `app.py` to include that feature in the input DataFrame before calling `scaler.transform`.

Troubleshooting

- Feature mismatch error: "The feature names should match those that were passed during fit." — This happens when the scaler or model was trained with a different set of features than the app sends. Retrain the model to include the same columns (or remove the extra columns from the app input).
- If anything else breaks, open `app.py` and the training script to ensure that `feature_names_in_` match between training and prediction.

If you want, I can:

- Add a `train_model.py` implementation and run it here to produce `scaler.pkl`/`student_grade_model.pkl` that include `attendance_percentage`.
- Add a lightweight CLI test harness (e.g., `predict_cli.py`) so you can test predictions from the terminal without Streamlit.

Contact

- If you'd like me to run the training and start the Streamlit app from this environment, say "Run training and start the app" and I'll proceed, run the training, verify artifacts, and launch Streamlit for you.
