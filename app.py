from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

#  Load Model 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "salary_pred.pkl")

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)
    
model = saved["model"] #ml model XGB Regressor
scaler = saved["scaler"] #scaler during training

#label encoders
job_title_encoder = saved["job_title_encoder"] 
education_level_encoder = saved["education_level_encoder"]
industry_encoder = saved["industry_encoder"]
company_size_encoder = saved["company_size_encoder"] 
location_encoder = saved["location_encoder"] 
remote_work_encoder = saved["remote_work_encoder"] 


#route

@app.route("/")
def home():
      return render_template("index.html",
                             job_titles=job_title_encoder.classes_,
                             education_levels=education_level_encoder.classes_,
                             industries=industry_encoder.classes_,
                             company_sizes=company_size_encoder.classes_,
                             locations=location_encoder.classes_,
                             remote_options=remote_work_encoder.classes_)


@app.route("/predict", methods=["POST"])
def predict():
      try:
            #Numerical inputs
            experience = float(request.form["experience"])
            skills = int(request.form["skills"])
            certifications= int(request.form["certifications"])
            
            #categorical inputs
            job_title = request.form['job_title']
            education_level = request.form['education_level']
            industry = request.form['industry']
            company_size = request.form['company_size']
            location = request.form['location']
            remote_work = request.form['remote_work']
            

            #encoding categorical values
            job_title_value = job_title_encoder.transform([job_title])[0]
            education_level_value = education_level_encoder.transform([education_level])[0]
            industry_value = industry_encoder.transform([industry])[0]
            company_size_value = company_size_encoder.transform([company_size])[0]
            location_value = location_encoder.transform([location])[0]
            remote_work_value = remote_work_encoder.transform([remote_work])[0]  



            # Prepare feature vector
            x = np.array([[job_title_value,experience,education_level_value,skills,industry_value,company_size_value,location_value,remote_work_value,certifications]])


            # Scale and predict
            x = scaler.transform(x)
            


            # salary prediction
            prediction = model.predict(x)[0]
            return render_template("result.html", prediction=f"₹{prediction:,.2f}")
            
      except Exception as e:
             return render_template("result.html", prediction=f"Error:{str(e)}")




if __name__ == '__main__':
    app.run(debug=True)