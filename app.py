import contractions
import re
import string
import nltk
from typing import List, Tuple

import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('wordnet')


# Define words lists
female_words_light = ['affectionate', 'agree', 'cheer', 'child',
                      'collaborate', 'collaboration',
                      'collaborative', 'collaboratively', 'community', 'compassion',
                      'compassionate', 'connect', 'considerate', 'cooperate',
                      'cooperation', 'cooperative', 'depend', 'dependable', 'emotional',
                      'empathetic', 'empathy', 'enthusiasm', 'enthusiast',
                      'enthusiastic', 'feel', 'flatterable', 'gentle',
                      'honest', 'inclusive']
female_words_medium = ['interdependent', 'interpersonal', 'intimate', 'kind',
                       'kinship', 'loyal', 'modest', 'modesty', 'nag', 'nurture',
                       'nurturing', 'patience', 'patient', 'pleasant', 'polite',
                       'quiet', 'sensitive', 'share', 'sharing', 'submissive',
                       'supportive', 'sympathetic', 'sympathy', 'tender',
                       'together', 'warm', 'whine', 'yield']
female_words_strong = ['female', 'females', 'woman', 'women',
                       'she', 'her', 'hers', 'maternity', 'ms', 'mrs', 'miss']

male_words_light = ['active', 'adventurous', 'aggressive', 'ambitious', 'assert',
                    'assertive', 'athletic', 'autonomous', 'autonomy', 'battle',
                    'boast', 'challenge', 'champion', 'compete', 'competitive',
                    'confident', 'courage', 'courageous', 'decision', 'decisive',
                    'defend', 'defensive', 'determine', 'direct', 'dominant',
                    'dominate', 'drive', 'driven', 'expert', 'fearless', 'fight',
                    'force', 'greedy', 'head-strong']
male_words_medium = ['headstrong', 'hierarchy',
                     'hostile', 'impulsive', 'independent', 'individual',
                     'independently', 'individually',
                     'intellect', 'lead', 'leader', 'logic', 'objective',
                     'opinion', 'outspoken', 'persist', 'principle', 'proactive', 'reckless',
                     'selfconfident', 'selfreliant', 'selfsufficient', 'strong',
                     'stubborn', 'superior', 'tackle', 'unreasonable']
male_words_strong = ['male', 'males', 'man',
                     'men', 'he', 'him', 'his', 'paternity', 'mr']

# Data preprocessing for strong word lists


def basic_text_cleaning(text):
    # Remove any HTML tags that might be present
    text = BeautifulSoup(text, "html.parser").get_text()

    # Add a space between words without spaces, except for all-uppercase words
    # Examples: ResponsibilitiesYour, ResponsibilitiesDuties
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove Emojis
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Expand Contractions (convert "can't" to "cannot")
    text = contractions.fix(text)

    # Remove Punctuation
    # "self-confident,!" = "selfconfident"
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text


def create_dataframe(text: str) -> pd.DataFrame:
    """
    Creates a DataFrame of preprocessed words from a given text string.

    Parameters:
        text (str): A string of text to preprocess.

    Returns:
        pd.DataFrame: A DataFrame of preprocessed words with columns 'original_word', 'cleaned_word_strong', and 'cleaned_word_medium_light'.
    """
    words = re.findall(r'\S+|\n', text)
    lemmatizer = WordNetLemmatizer()
    data = {'original_word': words,
            'cleaned_word_strong': [basic_text_cleaning(word) for word in words],
            'cleaned_word_medium_light': [lemmatizer.lemmatize(basic_text_cleaning(word)) for word in words]}
    return pd.DataFrame(data)


def highlight_words(df: pd.DataFrame, female_words: List[List[str]], male_words: List[List[str]]) -> str:
    """
    Highlights gendered terms in a DataFrame of preprocessed words based on their gender bias score.

    Parameters:
        df (pd.DataFrame): A DataFrame of preprocessed words with columns 'original_word', 'cleaned_word_strong', 'cleaned_word_medium_light', and 'cleaned_word_medium_light'.
        female_words (List[List[str]]): A list of female gendered terms divided into three levels of strength.
        male_words (List[List[str]]): A list of male gendered terms divided into three levels of strength.

    Returns:
        str: A string of words with gendered terms highlighted using HTML tags.
    """
    highlighted_text = []

    for _, row in df.iterrows():
        original_word = row['original_word']
        cleaned_word_strong = row['cleaned_word_strong']
        cleaned_word_medium = row['cleaned_word_medium_light']
        cleaned_word_light = row['cleaned_word_medium_light']

        if original_word == "\n":
            highlighted_text.append("<br>")
        elif cleaned_word_strong in female_words[2]:
            highlighted_text.append(
                f'<mark style="background-color: #FF4040;">{original_word}</mark>')
        elif cleaned_word_strong in male_words[2]:
            highlighted_text.append(
                f'<mark style="background-color: #4040FF;">{original_word}</mark>')
        elif cleaned_word_medium in female_words[1]:
            highlighted_text.append(
                f'<mark style="background-color: #FF8080;">{original_word}</mark>')
        elif cleaned_word_medium in male_words[1]:
            highlighted_text.append(
                f'<mark style="background-color: #8080FF;">{original_word}</mark>')
        elif cleaned_word_light in female_words[0]:
            highlighted_text.append(
                f'<mark style="background-color: #FFC0C0;">{original_word}</mark>')
        elif cleaned_word_light in male_words[0]:
            highlighted_text.append(
                f'<mark style="background-color: #C0C0FF;">{original_word}</mark>')
        else:
            highlighted_text.append(original_word)

    return ' '.join(highlighted_text)


def gender_bias_score(df: pd.DataFrame, female_words: List[List[str]], male_words: List[List[str]]) -> Tuple[int, int]:
    """
    Calculates the gender bias scores for female and male terms based on a DataFrame of preprocessed words.

    Parameters:
        df (pd.DataFrame): A DataFrame of preprocessed words with columns 'cleaned_word_strong', 'cleaned_word_medium_light', and 'cleaned_word_medium_light'.
        female_words (List[List[str]]): A list of female gendered terms divided into three levels of strength.
        male_words (List[List[str]]): A list of male gendered terms divided into three levels of strength.

    Returns:
        Tuple[int, int]: A tuple containing the gender bias scores for female and male terms, respectively.
    """
    female_score = 0
    male_score = 0

    for _, row in df.iterrows():
        cleaned_word_strong = row['cleaned_word_strong']
        cleaned_word_medium = row['cleaned_word_medium_light']
        cleaned_word_light = row['cleaned_word_medium_light']

        if cleaned_word_strong in female_words[2]:
            female_score += 5
        elif cleaned_word_strong in male_words[2]:
            male_score += 5
        elif cleaned_word_medium in female_words[1]:
            female_score += 2
        elif cleaned_word_medium in male_words[1]:
            male_score += 2
        elif cleaned_word_light in female_words[0]:
            female_score += 1
        elif cleaned_word_light in male_words[0]:
            male_score += 1

    return female_score, male_score


def plot_gender_bias_meter(female_score: int, male_score: int, gauge_number_color: str) -> Tuple[go.Figure, go.Figure]:
    """
    Plots two gauge charts to display the gender bias scores for female and male terms.

    Parameters:
        female_score (int): The gender bias score for female terms.
        male_score (int): The gender bias score for male terms.
        gauge_number_color (str): The color of the gauge chart values.

    Returns:
        Tuple[go.Figure, go.Figure]: A tuple containing the Plotly figures for the female and male gauge charts.
    """
    # Define the meter threshold values and labels
    threshold_values = [0, 5, 10, 100]
    threshold_labels = ['Mild', 'Moderate', 'Significant']

    # Create the gauge chart for female and male bias scores
    female_gauge = go.Indicator(
        mode="gauge+number",
        value=female_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Female"},
        number={'font': {'color': gauge_number_color}},
        gauge={
            'axis': {'range': [0, 20]},
            'bar': {'color': "rgba(255, 64, 64, 0.5)"},
            'steps': [
                {'range': [0, 5], 'color': "#FFD9D9"},
                {'range': [5, 10], 'color': "#FFB3B3"},
                {'range': [10, 20], 'color': "#FF8C8C"}],
        })

    male_gauge = go.Indicator(
        mode="gauge+number",
        value=male_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Male"},
        number={'font': {'color': gauge_number_color}},
        gauge={
            'axis': {'range': [0, 20]},
            'bar': {'color': "rgba(64, 64, 255, 0.5)"},
            'steps': [
                {'range': [0, 5], 'color': "#D9D9FF"},
                {'range': [5, 10], 'color': "#B3B3FF"},
                {'range': [10, 20], 'color': "#8C8CFF"}],
        })

    # Set the width of the gauge charts
    chart_width = 300

    # Customize the female gauge chart
    female_gauge_chart = go.Figure(female_gauge)
    female_gauge_chart.update_layout(
        width=chart_width,
        margin=dict(t=1, b=1, r=1, l=1),
        font=dict(color=gauge_number_color)
    )

    # Customize the male gauge chart
    male_gauge_chart = go.Figure(male_gauge)
    male_gauge_chart.update_layout(
        width=chart_width,
        margin=dict(t=1, b=1, r=1, l=1),
        font=dict(color=gauge_number_color)
    )

    return female_gauge_chart, male_gauge_chart


# Set the title of the Streamlit app
st.title("Gender Bias Analyzer")

# Increase the font size of the input label using markdown
st.markdown(
    '<style>label {font-size: 22px !important;}</style>', unsafe_allow_html=True)

# Create a text area for user input with a custom placeholder and styling
input_text = st.text_area("Enter Job Description to analyse:",
                          placeholder="Copy & Paste your Job Description here!", height=300, max_chars=None, key="input_text")

# Define a custom CSS style for the submit button
submit_button = st.markdown("""
<style>
    .custom_button {
        background-color: rgba(255, 165, 0, 0.3);
        border: none;
        padding: 8px 12px;
        font-size: 16px;
        cursor: pointer;
        text-align: center;
        margin-top: 5px;
        border-radius: 8px;
        box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .custom_button:hover {
        background-color: rgba(255, 165, 0, 0.5);
    }
</style>

<div style="text-align: center;">
    <input type='submit' value='Identify Gender Bias!' class='custom_button' onclick='window.location.reload();'>
</div>
""", unsafe_allow_html=True)

# If the submit button is clicked and the input text is not empty, perform analysis and display results
if submit_button:
    if input_text:
        # Call create_dataframe function to preprocess the input text and create a Pandas DataFrame
        df = create_dataframe(input_text)

        # Call gender_bias_score function to calculate the gender bias scores for female and male terms
        female_score, male_score = gender_bias_score(df, [female_words_light, female_words_medium, female_words_strong], [
                                                     male_words_light, male_words_medium, male_words_strong])

        # Call plot_gender_bias_meter function to create Plotly gauge charts for the gender bias scores
        female_gauge_chart, male_gauge_chart = plot_gender_bias_meter(
            female_score, male_score, 'black')

        # Display the gender bias meter and the analyzed text
        st.markdown("### Gender Bias Meter", unsafe_allow_html=True)
        st.markdown(
            '<style>h3 {margin-bottom: -50px !important;}</style>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(male_gauge_chart)
        with col2:
            st.plotly_chart(female_gauge_chart)
        st.markdown("### Analyzed Text", unsafe_allow_html=True)
        st.markdown(
            '<style>h3 {margin-top: 10px !important;}</style>', unsafe_allow_html=True)

        # Call highlight_words function to highlight gendered terms in the analyzed text
        highlighted_text = highlight_words(df, [female_words_light, female_words_medium, female_words_strong], [
                                           male_words_light, male_words_medium, male_words_strong])
        st.markdown(
            '<style>h3 {margin-top: 10px !important;}</style>', unsafe_allow_html=True)

        # Display the highlighted text in a styled box
        st.markdown(
            f'<div style="background-color: white; padding: 10px; border: 1px solid #ccc; white-space: pre-wrap;">{highlighted_text}</div>', unsafe_allow_html=True)
