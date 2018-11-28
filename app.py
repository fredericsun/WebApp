from flask import Flask, render_template
from flask_bootstrap import Bootstrap


app = Flask(__name__, static_url_path='/static')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
