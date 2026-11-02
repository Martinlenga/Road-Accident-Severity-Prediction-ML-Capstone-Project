# UK Road Safety Dashboard (2025)

A simple Streamlit dashboard for the three supplied UK Department for Transport road casualty CSV files.

## What the app does
- Filters collision records by date, severity, area type, speed limit and road type.
- Shows collision, casualty and vehicle counts.
- Charts collision trends, road conditions, casualty age/severity and driver age.
- Maps collision coordinates.
- Lets users download filtered collision records.

## 1. Install Python
Install Python 3.10 or newer from https://www.python.org/downloads/

## 2. Open this folder in VS Code
Keep the folder structure exactly like this:

```text
road_accident_streamlit_app/
├── app.py
├── requirements.txt
└── data/
    ├── dft-road-casualty-statistics-collision-2025.csv
    ├── dft-road-casualty-statistics-casualty-2025.csv
    └── dft-road-casualty-statistics-vehicle-2025.csv
```

## 3. Create and activate a virtual environment (Windows)
Open the VS Code terminal in this folder and run:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 4. Install packages

```bash
pip install -r requirements.txt
```

## 5. Run the app on your computer

```bash
streamlit run app.py
```

The app opens in your browser, usually at http://localhost:8501

## 6. Use the app
- Use the filters on the left to choose a date range, collision severity, area type, speed limit and road type.
- Open the tabs to explore Overview, Collision patterns, Casualties, Vehicles and Data.
- Use the download button in the Data tab to save filtered collision records.

## 7. Deploy it online with Streamlit Community Cloud
1. Create or sign in to a GitHub account: https://github.com/
2. Create a new repository, for example `uk-road-safety-dashboard`.
3. Upload `app.py`, `requirements.txt`, and the complete `data` folder. Keep the filenames and folder structure unchanged.
4. Make sure the repository is public if you are using the free Community Cloud workflow and are comfortable making the data public.
5. Go to https://share.streamlit.io/ and sign in with GitHub.
6. Choose **Create app** (or **Deploy an app**).
7. Select your repository, branch (usually `main`) and main file path `app.py`.
8. Click **Deploy** and wait for the build to finish.
9. Open the generated app URL and test the filters, tabs, map and download button.

If you update the code or data in GitHub, Community Cloud will redeploy the app.

## Notes
- The three files are related using `collision_index`. The app filters each table separately rather than joining all three raw tables, which avoids multiplying collision rows.
- The supplied files are for 2025. The app is not a live traffic feed.
- Some source columns contain numeric category codes. This starter app translates several common fields and leaves other codes visible as source codes.
- Do not add private data, passwords or API keys to a public GitHub repository.
