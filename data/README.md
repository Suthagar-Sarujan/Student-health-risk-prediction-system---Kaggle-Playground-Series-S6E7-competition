# Data directory

This project uses the "Student Health Risk" dataset from Kaggle
Playground Series S6E7. The raw CSVs are not included in this
repository (per Kaggle's terms of use) — download them yourself:

1. Go to the competition page on Kaggle and accept the rules.
2. Download `train.csv` (690,088 rows) and `test.csv` (295,753 rows).
3. Place both files directly in this `data/` folder:

    data/train.csv
    data/test.csv

Expected columns:
  id, sleep_duration, heart_rate, bmi, calorie_expenditure, step_count,
  exercise_duration, water_intake, diet_type, stress_level,
  sleep_quality, physical_activity_level, smoking_alcohol, gender,
  health_condition (target, train.csv only)
