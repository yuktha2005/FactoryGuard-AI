# TODO: Add Frontend UI for FactoryGuard AI

## Step 1: Prepare Model Files

- [x] Rename models/final_xgboost_model.pkl to models/factoryguard_xgb.pkl
- [x] Create models/feature_columns.pkl with the list of 40 feature names

## Step 2: Create Flask App

- [x] Create app.py by extracting and adapting the Flask code from Week4.ipynb, adding static file serving

## Step 3: Create Frontend Files

- [x] Create static/ directory
- [x] Create static/index.html: HTML form with inputs for all 40 features, submit button, and results display
- [x] Create static/script.js: JavaScript to handle form submission, send POST to /predict, display failure probability and top risk factors
- [x] Create static/style.css: Basic responsive styling for the UI

## Step 4: Update Requirements (if needed)

- [x] Check and add any missing dependencies to requirements.txt (Flask is already present)

## Step 5: Test the Application

- [ ] Run python app.py to start the Flask server on localhost:5000
- [ ] Open browser to localhost:5000 and test the UI with sample inputs
- [ ] Verify predictions and explanations are displayed correctly
