from flask import Flask, request, jsonify, render_template,render_template_string,send_file,redirect, url_for, session,flash
from flask_cors import CORS
import spacy
from spacy.matcher import Matcher
from models.resume import Resume
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
import nltk
nltk.download('stopwords')
from pyresparser import ResumeParser
import model1 as m
import model2 as m2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import datetime
from io import BytesIO
from zipfile import ZipFile
import PyPDF2 
import io 

app = Flask(__name__)
CORS(app)

app.secret_key = '1a2b3c4d5e6d7g8h9i10'

# MySQL configs
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'krsskrss4321'
app.config['MYSQL_DB'] = 'loginapp'


# Intialize MySQL
mysql = MySQL(app)

app.permanent_session_lifetime = datetime.timedelta(minutes=1)

spacy_model = spacy.load("en_core_web_sm")
matcher = Matcher(spacy_model.vocab)

yake_extractor = Resume.load_yake_extractor()

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['UPLOAD_FOLDER'] = 'uploads'
new_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], 'zip_files')
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)
    print(f"New folder zip_files created successfully!")

@app.route('/')
def index():
    if 'loggedin' in session:
        return render_template('home.html')
    return redirect(url_for('login'))

@app.route('/score')
def get_resume_score():
    return render_template('score.html')

@app.route('/resume_builder')
def get_resume():
    return render_template('resume_builder.html')

@app.route('/sort_resumes')
def sort_resumes():
    return render_template('sort_resumes.html')

@app.route('/process_files', methods=['POST'])
def process_files():
    if 'resume' not in request.files or 'job_description' not in request.files:
        return jsonify({"error": "Please upload both resume and job description files."})

    resume_file = request.files['resume']
    job_description_file = request.files['job_description']

    if resume_file.filename == '' or job_description_file.filename == '':
        return jsonify({"error": "Please select both resume and job description files."})
    if allowed_file(resume_file.filename) and allowed_file(job_description_file.filename):
        # Save uploaded files
        resume_filename = secure_filename(resume_file.filename)
        job_description_filename = secure_filename(job_description_file.filename)
        resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_filename)
        job_description_path = os.path.join(app.config['UPLOAD_FOLDER'], job_description_filename)
        resume_file.save(resume_path)
        job_description_file.save(job_description_path)

        # Extract text from PDF files
        resume_text = extract_text_from_pdf(resume_path)
        job_description_text = extract_text_from_pdf(job_description_path)
        resume = Resume(jd=job_description_text,res=resume_text)
        all_keywords, all_skills = resume.extract_all_job_keywords(yake_extractor=yake_extractor)
        # print(all_keywords)
        reskey=resume.extract_all_resume_keywords(yake_extractor=yake_extractor)
        resumekeywords,included_keywords, missing_keywords = resume.extract_included_and_missing_keywords(
            matcher=matcher, spacy_model=spacy_model ,yake_extractor=yake_extractor
        )
        if 'R' in all_keywords:
            all_keywords.remove('R')
        if 'R' in included_keywords:
            included_keywords.remove('R')
        if 'R' in all_skills:
            all_skills.remove('R')
        included_keywords=set(included_keywords)
        missing_keywords=set(missing_keywords)
        all_keywords=set(all_keywords)
        all_skills=set(all_skills)
        with open(resume_path, 'rb') as file:
            data = ResumeParser(file.name).get_extracted_data()
            print(file.name)
        score,keyscore=resume.get_resume_keyword_score(data)
        high_job_description=highlight_keywords(job_description_text,missing_keywords)

        #job role prediction
        pred=m.role(resume_text)
        print(pred)


        return render_template('scoreout.html',extracted_jd_words=all_keywords,common_keywords=included_keywords,job_description=high_job_description,score=(score*100),keyscore=int(keyscore*100),skills=all_skills,pred=pred)
    
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    jd=request.form['jd']
    if file and file.filename.endswith('.csv'):
        newname="res.csv"
        filename = os.path.join(app.config['UPLOAD_FOLDER'], newname)
        file.save(filename)
    df = pd.read_csv("uploads/res.csv", encoding='ISO-8859-1')
    new_column_names = ['name', 'resumes']
    df.columns = new_column_names
    df.to_csv("uploads/res.csv", index=False)
    m.process(jd)
    sortedcsv=pd.read_csv('sorted.csv', encoding='ISO-8859-1')
    return render_template('result.html',data=sortedcsv.to_html())

@app.route('/download', methods=['GET'])
def download_file():
    # Assuming your CSV file name is 'output.csv' in the 'uploads' folder
    file_path = 'sorted.csv'

    # Send the CSV file to the user for download
    return send_file(file_path, as_attachment=True)    

@app.route('/upload_res', methods=['POST'])
def upload_file_res():
    if 'file' not in request.files:
        return "No file part"
    scores = {}
    file = request.files['file']
    jd = request.form.get("jd")

    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return "No selected file"
    
        # Extract PDFs from the uploaded zip file
    with ZipFile(file, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
            # Check if the file is a PDF
                if file_name.lower().endswith('.pdf'):
                # Open the PDF file
                    with zip_ref.open(file_name) as pdf_file:
                        # Read the PDF content
                        pdf_data = pdf_file.read()
                        # Extract text from the PDF
                        text = extract_text_from_pdf_bytes(pdf_data)
                        # Print the extracted text
                        print(f"Text from {file_name}:")
                        # print(text)
                        print("=" * 50)
                        # print(f"new folder path {new_folder_path}")
                        pdf_path = os.path.join(new_folder_path, file_name.split('/')[-1])
                        print(f"pdf path {pdf_path}")
                        with open(pdf_path, 'wb') as pdf_output_file:
                            pdf_output_file.write(pdf_data)
                        # vectorizer = TfidfVectorizer(max_features=500)
                        # tfidf_matrix = vectorizer.fit_transform([text, jd])
                        # cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
                        # similarity_score = cosine_sim[0][0]
                        resume = Resume(jd=jd,res=text)
                        all_keywords, all_skills = resume.extract_all_job_keywords(yake_extractor=yake_extractor)
                        # print(all_keywords)
                        reskey=resume.extract_all_resume_keywords(yake_extractor=yake_extractor)
                        resumekeywords,included_keywords, missing_keywords = resume.extract_included_and_missing_keywords(
                            matcher=matcher, spacy_model=spacy_model ,yake_extractor=yake_extractor
                        )
                        
                        print('================================================')
                        with open(pdf_path, 'rb') as file:
                            data = ResumeParser(file.name).get_extracted_data()
                            print(data)
                            print("=============================================")
                        score,keyscore=resume.get_resume_keyword_score(data)
                        scores[file_name] = keyscore
                        print(keyscore)
            print(scores)
            sorted_pdfs = sort_pdfs(scores)
            print('after returning')
            print(sorted_pdfs)

            # Create a new zip file with sorted PDFs
            sorted_zip_bytes = BytesIO()
            counter=1
            with ZipFile(sorted_zip_bytes, "w") as sorted_zip:
                for pdf in sorted_pdfs:
                    with zip_ref.open(pdf) as pdf_file:
                        filename = f"{counter}_{pdf.split('/')[-1]}"
                        sorted_zip.writestr(filename, pdf_file.read())
                        counter+=1
            sorted_zip_bytes.seek(0)
    # Send the sorted zip file back to the client
    return send_file(
        io.BytesIO(sorted_zip_bytes.getvalue()), mimetype="application/zip", download_name="sorted.zip"
    )
    

def generate_scores():
    pass 

def sort_pdfs(scores):
    sorted_pdfs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [pdf for pdf, _ in sorted_pdfs]

def highlight_keywords(job_description, missing_keywords):
    highlighted_description = job_description
    for keyword in missing_keywords:
        highlighted_description = highlighted_description.replace(keyword, f'<span class="highlight">{keyword}</span>')
    return highlighted_description

def extract_text_from_pdf_bytes(pdf_data):
    # Function to extract text from PDF data using PyPDF2
    pdf_stream = io.BytesIO(pdf_data)  # Wrap the PDF data in a BytesIO object
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text.strip()

def extract_text_from_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    pdf_text = ''
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password))
        
        account = cursor.fetchone()
                
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            
            return redirect(url_for('index'))
        else:
            flash("Incorrect username/password!", "danger")
    return render_template('auth/login.html',title="Login")


@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
                
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute( "SELECT * FROM accounts WHERE username LIKE %s", [username] )
        account = cursor.fetchone()
    
        if account:
            flash("Account already exists!", "danger")
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash("Invalid email address!", "danger")
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash("Username must contain only characters and numbers!", "danger")
        elif not username or not password or not email:
            flash("Incorrect username/password!", "danger")
        else:
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username,email, password))
            mysql.connection.commit()
            flash("You have successfully registered!", "success")
            return redirect(url_for('login'))

    elif request.method == 'POST':
        flash("Please fill out the form!", "danger")
    return render_template('auth/register.html',title="Register")


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/profile')
def profile():
    if 'loggedin' in session:
       
        return render_template('auth/profile.html', username=session['username'],title="Profile")
    return redirect(url_for('login'))  


if __name__ == '__main__':
    app.run(debug=True)



