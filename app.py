import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your pre-trained model
model = load_model('imageclassification231.keras')
data_cat = ['barbell', 'dumbbell', 'gym_ball', 'kettlebell', 'smith_machine']  # Update with your actual classes

# Mapping from equipment to exercises and links (use your existing dictionaries)
machine_exercises = {
        'barbell': {
        'upper_body': ['Bench Press', 'Barbell Rows', 'Overhead Press'],
        'lower_body': ['Squats', 'Deadlifts', 'Lunges'],
        'core': ['Russian Twists', 'Barbell Rollouts', 'Hanging Leg Raises']
    },
    'dumbbell': {
        'upper_body': ['Dumbbell Chest Press', 'Shoulder Press', 'Dumbbell Rows'],
        'lower_body': ['Dumbbell Squats', 'Dumbbell Romanian Deadlifts', 'Step-ups'],
        'core': ['Dumbbell Side Bends', 'Dumbbell Woodchoppers', 'Plank Rows']
    },
    'gym_ball': {
        'core': ['Ball Crunches', 'Plank with Leg Lifts', 'Russian Twists']
    },
    'kettlebell': {
        'upper_body': ['Kettlebell Swings', 'Kettlebell Press', 'Kettlebell Rows'],
        'lower_body': ['Kettlebell Goblet Squats', 'Kettlebell Lunges', 'Kettlebell Deadlifts'],
        'core': ['Kettlebell Russian Twists', 'Kettlebell Windmills', 'Kettlebell Turkish Get-ups']
    },
    'smith_machine': {
        'upper_body': ['Smith Machine Bench Press', 'Smith Machine Shoulder Press', 'Smith Machine Rows'],
        'lower_body': ['Smith Machine Squats', 'Smith Machine Lunges', 'Smith Machine Deadlifts']
    }
    
    }  # your existing dictionary
exercise_links = {
        'Bench Press': 'https://www.youtube.com/shorts/0cXAp6WhSj4',
    'Barbell Rows': 'https://www.youtube.com/shorts/Nqh7q3zDCoQ',
    'Overhead Press': 'https://www.youtube.com/shorts/zSU7T1zZagQ',
    'Squats': 'https://www.youtube.com/shorts/gslEzVggur8',
    'Deadlifts': 'https://www.youtube.com/shorts/8np3vKDBJfc',
    'Lunges': 'https://www.youtube.com/shorts/TwEH620Pn6A',
    'Russian Twists': 'https://www.youtube.com/watch?v=Tau0hsW8iR0',
    'Barbell Rollouts': 'https://www.youtube.com/watch?v=3C1TRMJveXo',
    'Hanging Leg Raises': 'https://www.youtube.com/shorts/2n4UqRIJyk4',
    'Dumbbell Chest Press': 'https://www.youtube.com/shorts/SidmT09GXz8',
    'Shoulder Press': 'https://www.youtube.com/shorts/dyv6g4xBFGU',
    'Dumbbell Rows': 'https://www.youtube.com/shorts/s1H87k4tAaA',
    'Dumbbell Squats': 'https://www.youtube.com/shorts/eLX_dyvooKQ',
    'Dumbbell Romanian Deadlifts': 'https://www.youtube.com/shorts/u14AwrUcwWw',
    'Step-ups': 'https://www.youtube.com/shorts/PzDbmqL6qo8Dumbbell',
    'Dumbbell Side Bends': 'https://www.youtube.com/watch?v=dL9ZzqtQI5c', 
    'Dumbbell Woodchoppers': 'https://www.youtube.com/shorts/OgQU_bbdB7c',
    'Plank Rows': 'https://www.youtube.com/watch?v=Gtc_Ns3qYYo',
    'Ball Crunches': 'https://www.youtube.com/watch?v=O4d3kd1ZLyc',
    'Plank with Leg Lifts': 'https://www.youtube.com/shorts/s_8PheAKUYk',
    'Kettlebell Swings': 'https://www.youtube.com/shorts/SR_4kUbkEaw',
    'Kettlebell Press': 'https://www.youtube.com/watch?v=eKQ0JOx_1qI',
    'Kettlebell Rows': 'https://www.youtube.com/shorts/e4OSLk1qZOQ',
    'Kettlebell Goblet Squats': 'https://www.youtube.com/shorts/dBnNCOtuGNQ',
    'Kettlebell Lunges': 'https://www.youtube.com/shorts/otd2YQk7osI',
    'Kettlebell Deadlifts': 'https://www.youtube.com/shorts/I7q_EPywprs',
    'Kettlebell Russian Twists': 'https://www.youtube.com/shorts/BA-uP_-bVE8',
    'Kettlebell Windmills': 'https://www.youtube.com/shorts/OVNXkKsfy7o',
    'Kettlebell Turkish Get-ups': 'https://www.youtube.com/shorts/-dfk79o6iHI',
    'Smith Machine Bench Press': 'https://www.youtube.com/shorts/G-jT0m0nvx8',
    'Smith Machine Shoulder Press': 'https://www.youtube.com/shorts/QWdaC7rQ-FM',
    'Smith Machine Rows': 'https://www.youtube.com/shorts/qivPkcDI0s0',
    'Smith Machine Squats': 'https://www.youtube.com/shorts/xU4cuTffVZc',
    'Smith Machine Lunges': 'https://www.youtube.com/shorts/dFMa-mmZ6A8',
    'Smith Machine Deadlifts': 'https://www.youtube.com/shorts/Bhg9IvQzsCI'
    
    }     # your existing dictionary

def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch)
    score = tf.nn.softmax(predictions[0])
    return data_cat[np.argmax(score)], np.max(score)

def suggest_exercises(equipment):
    if equipment not in machine_exercises:
        return "This equipment is not available for suggestions."
    
    response = f"Suggested exercises for {equipment}:\n"
    for part, exercises in machine_exercises[equipment].items():
        response += f"\n{part.capitalize()} exercises:\n"
        for exercise in exercises:
            link = exercise_links.get(exercise, "No link available")
            response += f"- {exercise}: [Watch here]({link})\n"
    return response

# Streamlit UI components
st.title("Exercise Suggestion App")

uploaded_file = st.file_uploader("Choose an image of exercise equipment", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    # Save the uploaded image to a path
    with open("dataset/user_upload/uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully.")

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Predict and display the results
    equipment, confidence = predict_image("dataset/user_upload/uploaded_image.jpg")
    st.write(f"Predicted Equipment: {equipment} with a confidence of {confidence*100:.2f}%")

    # Suggest exercises
    suggestions = suggest_exercises(equipment)
    st.markdown(suggestions)

