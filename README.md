# Flight Price Prediction ✈️

## Overview
This project predicts flight ticket prices based on various
parameters such as airline, source, destination, duration,
and stops. Built using Python and deployed as a web
application using Flask.

## Project Structure
Flight_Price_Prediction/
├── Data_Train.xlsx              # Main training dataset
├── Data_Train_Additional.xlsx   # Additional training data
├── Test_set.xlsx                # Test dataset
├── Test_set_Additional.xlsx     # Additional test data
├── app.py                       # Flask web application
├── flight_model_v2.py           # ML Model script
├── skyfare_v2.html              # Frontend HTML page
└── README.md                    # Project documentation

## Features
- Predicts flight ticket prices accurately
- Interactive web interface
- Trained on real flight data
- Multiple airline and route support
- Easy to use form based input

## Input Parameters Used
- Airline Name
- Source City
- Destination City
- Date of Journey
- Departure Time
- Arrival Time
- Duration of Flight
- Number of Stops
- Additional Info

## ML Algorithms Used
- Random Forest Regressor
- Decision Tree Regressor
- Linear Regression
- Best model selected based on accuracy

## Tech Stack
| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data processing |
| Scikit-learn | ML algorithms |
| Flask | Web framework |
| HTML/CSS | Frontend (skyfare_v2.html) |
| Excel (XLSX) | Dataset storage |

## How to Run the Project

### Step 1 - Clone the repository
git clone https://github.com/Sudharshan2112/Flight_Price_Prediction.git

### Step 2 - Install required libraries
pip install flask pandas scikit-learn numpy openpyxl

### Step 3 - Run the Flask app
python app.py

### Step 4 - Open in browser
http://localhost:5000

## Dataset
- Training Data: Data_Train.xlsx & Data_Train_Additional.xlsx
- Test Data: Test_set.xlsx & Test_set_Additional.xlsx
- Contains real flight price data with multiple airlines

## Results
- Model Accuracy: XX%
- Best Algorithm: Random Forest Regressor
- Mean Absolute Error: XX

## Airlines Covered
- IndiGo
- Air India
- Jet Airways
- SpiceJet
- Vistara
- GoAir
- Multiple Charter Airlines

## Routes Covered
- Bangalore → Delhi
- Mumbai → Hyderabad
- Delhi → Cochin
- And many more routes!

## Author
**Sudharshan NS**
- GitHub: Sudharshan2112

## License
This project is open source and available for learning purposes.
