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
female_explicit_biased_words = ['female', 'females', 'woman', 'women', 'she',
                                'her', 'hers', 'maternity', 'mrs', 'miss']

weakly_feminine_words = ['affectionate', 'approachable', 'attune', 'calming',
                         'cheer', 'collaborate', 'collaboratively', 'connect',
                         'cooperate', 'cooperation', 'cooperative', 'dedicated',
                         'dependable', 'diplomatic', 'expressive', 'heartfelt',
                         'inclusive', 'interdependent', 'interpersonal', 'listening',
                         'loyal', 'maternity', 'mediating', 'mindful', 'modest',
                         'organized']

strongly_feminine_words = ['attentive', 'caring', 'collaborative', 'community',
                           'compassion', 'compassionate', 'considerate', 'creative',
                           'emotional', 'empathetic', 'empathy', 'encouraging',
                           'enthusiastic', 'friendly', 'gentle', 'harmonious',
                           'helpful', 'honest', 'humble', 'inspiring', 'kind',
                           'nurturing', 'patient', 'supportive']

male_explicit_biased_words = ['male', 'males', 'man', 'men', 'he',
                              'him', 'his', 'paternity', 'mr']

weakly_masculine_words = ['assertive', 'athletic', 'authoritative', 'autonomy',
                          'autonomous', 'battle', 'boast', 'commanding', 'courageous',
                          'defensive', 'determine', 'enterprising', 'exceed',
                          'exceptional', 'grit', 'headstrong', 'hierarchical',
                          'individually', 'industrious', 'logical', 'masterful',
                          'motivate', 'negotiate', 'outgoing', 'outspoken',
                          'resilient', 'strategic']

strongly_masculine_words = ['aggressive', 'ambitious', 'bold', 'challenge', 'competitive',
                            'confident', 'decisive', 'defend', 'direct', 'dominant',
                            'disciplined', 'driven', 'energetic', 'fight', 'force',
                            'focused', 'independent', 'influential', 'lead', 'powerful',
                            'resolve', 'robust', 'strong']

# Data preprocessing for strong word lists


def basic_text_cleaning(text):
    # Remove any HTML tags that might be present
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove Emojis
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Add a space between words without spaces, except for all-uppercase words
    # Examples: ResponsibilitiesYour, ResponsibilitiesDuties
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    # Convert text to lowercase
    text = text.lower()

    # Replace punctuation with a space
    text = re.sub(r'['+string.punctuation+']', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

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
            'cleaned_word': [basic_text_cleaning(word) for word in words],
            'further_cleaned_word': [lemmatizer.lemmatize(basic_text_cleaning(word)) for word in words]}
    return pd.DataFrame(data)


def highlight_words(df: pd.DataFrame, female_explicit_biased_words: List[str], male_explicit_biased_words: List[str],
                   strongly_feminine_words: List[str], strongly_masculine_words: List[str],
                   weakly_feminine_words: List[str], weakly_masculine_words: List[str]) -> str:
    """
    Highlights gendered terms in the input text with different colors based on their gendered strength.

    Parameters:
        df (pd.DataFrame): A DataFrame of preprocessed words with columns 'original_word', 'cleaned_word', and 'further_cleaned_word'.
        female_explicit_biased_words (List[str]): A list of explicitly biased female terms.
        male_explicit_biased_words (List[str]): A list of explicitly biased male terms.
        strongly_feminine_words (List[str]): A list of strongly gendered female terms.
        strongly_masculine_words (List[str]): A list of strongly gendered male terms.
        weakly_feminine_words (List[str]): A list of weakly gendered female terms.
        weakly_masculine_words (List[str]): A list of weakly gendered male terms.

    Returns:
        str: The input text with gendered terms highlighted using different colors.
    """
    highlighted_text = []
    for _, row in df.iterrows():
        original_word = row['original_word']
        cleaned_word = row['cleaned_word']
        further_cleaned_word = row['further_cleaned_word']

        if original_word == "\n":
            highlighted_text.append("<br>")
        elif cleaned_word in female_explicit_biased_words:
            highlighted_text.append(
                f'<mark style="background-color: #FF4040;">{original_word}</mark>')
        elif cleaned_word in male_explicit_biased_words:
            highlighted_text.append(
                f'<mark style="background-color: #4040FF;">{original_word}</mark>')
        elif further_cleaned_word in strongly_feminine_words:
            highlighted_text.append(
                f'<mark style="background-color: #FFA0A0;">{original_word}</mark>')
        elif further_cleaned_word in strongly_masculine_words:
            highlighted_text.append(
                f'<mark style="background-color: #A0A0FF;">{original_word}</mark>')
        elif further_cleaned_word in weakly_feminine_words:
            highlighted_text.append(
                f'<mark style="background-color: #FFD0D0;">{original_word}</mark>')
        elif further_cleaned_word in weakly_masculine_words:
            highlighted_text.append(
                f'<mark style="background-color: #D0D0FF;">{original_word}</mark>')
        else:
            highlighted_text.append(original_word)

    return ' '.join(highlighted_text)



def gender_bias_score(df: pd.DataFrame, strongly_feminine_words: List[str], strongly_masculine_words: List[str],
                      weakly_feminine_words: List[str], weakly_masculine_words: List[str]) -> Tuple[int, int]:
    """
    Calculates the gender bias scores for female and male terms based on a DataFrame of preprocessed words.

    Parameters:
        df (pd.DataFrame): A DataFrame of preprocessed words with columns 'further_cleaned_word'.
        strongly_feminine_words (List[str]): A list of female gendered terms with a score of 2.
        strongly_masculine_words (List[str]): A list of male gendered terms with a score of 2.
        weakly_feminine_words (List[str]): A list of female gendered terms with a score of 1.
        weakly_masculine_words (List[str]): A list of male gendered terms with a score of 1.

    Returns:
        Tuple[int, int]: A tuple containing the gender bias scores for female and male terms, respectively.
    """
    female_score = 0
    male_score = 0

    for _, row in df.iterrows():
        further_cleaned_word = row['further_cleaned_word']

        if further_cleaned_word in strongly_feminine_words:
            female_score += 2
        elif further_cleaned_word in strongly_masculine_words:
            male_score += 2
        elif further_cleaned_word in weakly_feminine_words:
            female_score += 1
        elif further_cleaned_word in weakly_masculine_words:
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
st.title("Gender Bias Text Analyzer")

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
        female_score, male_score = gender_bias_score(
            df,
            strongly_feminine_words, strongly_masculine_words,
            weakly_feminine_words, weakly_masculine_words
        )

        # Call plot_gender_bias_meter function to create Plotly gauge charts for the gender bias scores
        female_gauge_chart, male_gauge_chart = plot_gender_bias_meter(
            female_score, male_score, 'black')

        explicit_bias_detected = any(word in df['cleaned_word'].tolist() for word in female_explicit_biased_words + male_explicit_biased_words)

        # Define a custom CSS style for the bias message box
        st.markdown("""
        <style>
            .bias_message_box {
                background-color: rgba(240, 240, 240, 1);
                padding: 8px 12px;
                font-size: 22px;
                text-align: center;
                margin-top: 0px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100%;
            }
            .bias_message_box.bias_message_box_male {
                background-color: rgba(192, 192, 255, 0.3);
            }
            .bias_message_box.bias_message_box_female {
                background-color: rgba(255, 192, 192, 0.3);
            }
        </style>
        """, unsafe_allow_html=True)

        if explicit_bias_detected:
            st.markdown('<style>h3 {margin-top: 50px !important;}</style>', unsafe_allow_html=True)
            st.markdown('<p style="text-align: center; color: darkred; font-size: 30px; font-weight: bold;">Explicit Bias Detected!</p>', unsafe_allow_html=True)

        # Display the gender bias meter and the analyzed text
        st.markdown("### Gender Bias Meter", unsafe_allow_html=True)

        st.markdown(
            '<style>h3 {margin-bottom: -50px !important;}</style>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(male_gauge_chart)
        with col2:
            st.plotly_chart(female_gauge_chart)

        score_diff = male_score - female_score
        if any(word in df['cleaned_word'].tolist() for word in male_explicit_biased_words):
            bias_message_box_class = "bias_message_box bias_message_box_male"
            bias_message = "Job Description is <b>explicitly</b> biased towards Male Candidates"
        elif any(word in df['cleaned_word'].tolist() for word in female_explicit_biased_words):
            bias_message_box_class = "bias_message_box bias_message_box_female"
            bias_message = "Job Description is <b>explicitly</b> biased towards Female Candidates"
        elif abs(score_diff) < 5:
            bias_message_box_class = "bias_message_box"
            bias_message = "Job Description is Gender Neutral"
        elif 5 <= abs(score_diff) <= 10:
            bias_message_box_class = "bias_message_box"
            bias_message = f"Job Description is <b>slightly</b> biased towards {'Male' if score_diff > 0 else 'Female'} Candidates"
        elif 10 < abs(score_diff) <= 20:
            bias_message_box_class = "bias_message_box"
            bias_message = f"Job Description is <b>more</b> biased towards {'Male' if score_diff > 0 else 'Female'} Candidates"
        else:
            bias_message_box_class = "bias_message_box"
            bias_message = f"Job Description is <b>extremely</b> biased towards {'Male' if score_diff > 0 else 'Female'} Candidates"

        st.markdown(f'<div class="{bias_message_box_class}"><span>{bias_message}</span></div>', unsafe_allow_html=True)

        st.markdown("### Analyzed Job Description", unsafe_allow_html=True)
        st.markdown(
            '<style>h3 {margin-top: 10px !important;}</style>', unsafe_allow_html=True)

        # Call highlight_words function to highlight gendered terms in the analyzed text
        highlighted_text = highlight_words(
            df,
            female_explicit_biased_words, male_explicit_biased_words,
            strongly_feminine_words, strongly_masculine_words,
            weakly_feminine_words, weakly_masculine_words
        )
        st.markdown(
            '<style>h3 {margin-top: 10px !important;}</style>', unsafe_allow_html=True)

        # Display the highlighted text in a styled box
        st.markdown(
            f'<div style="background-color: #f5f5f5; padding: 20px; border: 1px solid #ccc; white-space: pre-wrap; border-radius: 10px; line-height: 1.6;">{highlighted_text}</div>', unsafe_allow_html=True)
