from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("boston_model.pkl", "rb"))


# HOME ROUTE (GET ONLY)
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction_text=None)


# PREDICTION ROUTE (POST ONLY)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form inputs in correct order
        features = [
            float(request.form["crim"]),
            float(request.form["zn"]),
            float(request.form["indus"]),
            float(request.form["chas"]),
            float(request.form["nox"]),
            float(request.form["rm"]),
            float(request.form["age"]),
            float(request.form["dis"]),
            float(request.form["rad"]),
            float(request.form["tax"]),
            float(request.form["ptratio"]),
            float(request.form["b"]),
            float(request.form["lstat"])
        ]

        # Convert to numpy array
        final_input = np.array(features).reshape(1, -1)

        # Predict price
        prediction = model.predict(final_input)[0]

        output_text = f"Predicted House Price: ${prediction:.2f}K"

        return render_template(
            "index.html",
            prediction_text=output_text
        )

    # If user enters letters / invalid values
    except ValueError:
        return render_template(
            "index.html",
            prediction_text="Please enter numeric values only."
        )

    # Any other error
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


# RUN APP
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
