from flask import Flask, request
from flask import render_template

from question_gen import generate_questions

app = Flask(__name__)

@app.route('/input')
def input_page():
    return render_template('user_prompt.html')

@app.route('/quiz', methods = ['POST', 'GET'])
def start_quiz():
    if request.method == 'POST':
        quiz_material = request.form['quiz_material']
        #qa_pairs = [("How many sides does a square have?", "4"), ("What is the color of a banana?", "yellow"), ("Who created Microsoft?", "Bill Gates")]
        qa_pairs = generate_questions(quiz_material)
        return render_template('solve_quiz.html', qa_pairs = qa_pairs)
		#return quiz_material
    else:
        return 'GET!'
