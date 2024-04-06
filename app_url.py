# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:56:17 2024

@author: vinod
"""

from flask import Flask, render_template, request, redirect, url_for, session
from model_url import user_pref, recommend_books_similar_to 

app = Flask(__name__)
app.secret_key = '123'

#@app.route("/")
@app.route('/', methods=['Get'])
def main():
    return render_template('index_new.html')

@app.route("/quiz", methods = ['Get'])
def quiz():
    return render_template("quiz.html")

@app.route("/similar", methods = ['Get'])
def similar():
    return render_template("similar.html")

@app.route("/quiz_submit", methods = ['Post'])
def quiz_input():
    genres = request.form['genres']
    themes = request.form['themes']
    settings = request.form['settings']
    development = request.form['development']
    pace = request.form['pace']
    top_books_list, list_summary, links = user_pref(genres, themes, settings, development, pace)
    return render_template("quiz_outputurl.html", top_books=top_books_list, list_summary=list_summary, book_urls = links)

@app.route("/similar_submit", methods = ['Post'])
def similar_input():
    book_name = request.form['genres']
    book_name = book_name.lower()
    recommended_books, list_of_summary,links = recommend_books_similar_to(book_name)
    return render_template("quiz_outputurl.html", top_books = recommended_books, list_summary = list_of_summary, book_urls = links)


if __name__=='__main__':
    app.run(host='localhost',port=8000)



