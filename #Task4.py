import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("RTA Dataset.csv")

# Function to clean time data
def clean_time(time_str):
    if pd.isna(time_str) or time_str.strip() == '':
        return '00:00:00'
    parts = time_str.split(':')
    if len(parts) == 2:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"
    if len(parts) == 3:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
    return time_str

# Apply the cleaning function to the Time column
df['Time'] = df['Time'].apply(clean_time)

# Convert Time column to datetime and extract hour
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour

# Verify the conversion
print(df['Time'].head())

# Check for missing values and handle them
df = df.dropna()  

# Convert categorical columns to category type
categorical_cols = ['Day_of_week', 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 
                    'Vehicle_driver_relation', 'Driving_experience', 'Type_of_vehicle', 
                    'Owner_of_vehicle', 'Service_year_of_vehicle', 'Defect_of_vehicle', 
                    'Area_accident_occured', 'Lanes_or_Medians', 'Road_allignment', 
                    'Types_of_Junction', 'Road_surface_type', 'Road_surface_conditions', 
                    'Light_conditions', 'Weather_conditions', 'Type_of_collision', 
                    'Vehicle_movement', 'Casualty_class', 'Sex_of_casualty', 'Age_band_of_casualty', 
                    'Casualty_severity', 'Work_of_casuality', 'Fitness_of_casuality', 
                    'Pedestrian_movement', 'Cause_of_accident', 'Accident_severity']

for col in categorical_cols:
    df[col] = df[col].astype('category')

# Summary statistics
print(df.describe(include='all'))

# Plotting
plt.figure(figsize=(12, 6))
sns.countplot(x='Time', data=df, palette='viridis')
plt.title('Accident Frequency by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Day_of_week', data=df, palette='viridis')
plt.title('Accident Frequency by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Accidents')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Weather_conditions', data=df, palette='viridis')
plt.title('Accident Frequency by Weather Conditions')
plt.xlabel('Weather Conditions')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Road_surface_conditions', data=df, palette='viridis')
plt.title('Accident Frequency by Road Surface Conditions')
plt.xlabel('Road Surface Conditions')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

# Encode categorical variables if needed for further analysis
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Check the first few rows of the dataframe
print(df.head())




