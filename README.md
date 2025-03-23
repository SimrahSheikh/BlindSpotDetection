# Blind Spot Detection System (Sensor Fusion + AI)

##  Project Overview  
This project aims to develop a **Blind Spot Detection System** using **LiDAR and Radar sensor fusion** in **adverse weather conditions** such as **fog and rain**. The system improves **object detection accuracy** using **Kalman Filtering** and generates datasets for AI-based predictive analysis. The project also includes a **machine learning model trained using XGBoost and Random Forest** to classify collision risks.

**1. Sensor Fusion using MATLAB**
- Simulated **LiDAR and Radar** sensors for blind spot detection.
- Implemented **Kalman Filtering** to improve object tracking accuracy.
- **Generated CSV dataset** for AI model training. 

 **2. AI-Based Collision Prediction**
- Used **XGBoost and Random Forest** to train an AI model on sensor data.
- The AI model **predicts collision risk** based on: **Object Distance**, **Velocity (Relative Speed)**, **Movement Trajectory**, **Vehicle Speed**
- **Trained on MATLAB-generated sensor data** and evaluated using ML techniques.

Project Structure -

Sensor Fusion using MATLAB file is :blindspot.m
Sample Output Fugures and CSV file are also attached: 1)sensor_fusion_output.csv outpout csv   2)Figure 1,2,4 containg Plot of sensor output.

AI model : main.py




