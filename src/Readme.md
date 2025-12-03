## src folder


```text
src
├── Data
│   ├── Html
│   │   ├── May 2025 Breakfast production records/
│   │   │   (all daily *.html breakfast files)
│   │   └── May 2025 Lunch production records/
│   │       (all daily *.html lunch files)
│   │
│   └── Output
│       ├── breakfast_combined.csv
│       ├── lunch_combined.csv
│       └── meals_combined.csv
│
├── component
│   ├── EDA.py
│   ├── preprocess.py
│   │
│   ├── multivariate
│   │   ├── model.py
│   │   ├── plot.py
│   │   ├── training.py
│   │   └── utils.py
│   │
│   └── univariate
│       ├── comparing_model.py
│       ├── forecasting.py
│       ├── model.py
│       ├── plot.py
│       ├── training.py
│       └── utils.py
│
├── maincode
│   └── main.py
│
├── results
│   └── all_school_meal_forecasts.csv
│
├── tests
│    ├── combine_csv.py
│    └── multivariate_main.py
│   
│
├── .gitkeep
└── requirements.txt
```


🚀 How to Run
To execute the full pipeline and generate fresh outputs for the dashboard:

Ensure prerequisites are installed (see root README).

Run the Main :
```bash
python src/maincode/main.py
```

(Note: Adjust the path based on your current working directory)
