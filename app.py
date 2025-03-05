from flask import Flask, render_template, request, redirect, url_for
from model import recommend, get_persona

app = Flask(__name__)

@app.route('/')
def persona():
    """Render the persona selection page."""
    return render_template('persona.html')

@app.route('/info2.html')
def info2():
    place_id = request.args.get('id')  # This will fetch the 'id' parameter from the URL
    if place_id:
        # Handle logic to display the destination info based on the ID
        return render_template('info2.html', place_id=place_id)
    else:
        return "Destttttination not found", 404


@app.route('/home', methods=['POST'])
def home():
    """Handle form submission, run recommendations, and display results."""
    preferred_subtype = request.form.get('subtype')
    keywords = request.form.get('keywords').lower().split(',')
    preferred_city = request.form.get('city')

    # Clean keywords
    keywords = [kw.strip() for kw in keywords if kw.strip()]

    # Get recommendations
    recommendations = recommend(preferred_subtype, keywords, preferred_city)

    return render_template('recommendation.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
