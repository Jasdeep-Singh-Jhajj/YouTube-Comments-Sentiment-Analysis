"""
This is main module to run a Flask web framework to input a YouTube video URL and
display the Sentiment Analysis using SVM model results
Reference: https://www.geeksforgeeks.org/flask-creating-first-simple-application/
"""
import shutil
import os
from flask import Flask, render_template, request
from yt_comment_analysis import analyze_video

app = Flask(__name__)

def copy_images_to_static():
    source_dir = '../images'
    destination_dir = 'static'
    
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    print(source_dir)
    print(destination_dir)
    
    # Copy files from source to destination
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        video_url = request.form['video_url']
        youtube_api_key = 'AIzaSyCVRR-N5-Fa1WE9maQ4uMcqXtUEhF-ven4'
        title, accuracy, best_params = analyze_video(youtube_api_key, video_url)
        copy_images_to_static()

        return render_template('results.html', title=title, results=accuracy,
                               best_params=best_params)


if __name__ == '__main__':
    app.run(debug=True)
