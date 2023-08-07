from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid 
app= Flask(__name__)

@app.route('/',methods=['GET','POST'])
def hello_world():
    request_type_str= request.method
    if request_type_str == 'GET':
        return render_template('index.html', href='static/base_pic.svg')
    else:
        text= request.form['text']
        random_string=uuid.uuid4().hex
        path="static/"+ random_string+".svg"
        model_in=load('model.joblib')
        np_arr=floats_string_to_np_arr(text)
        make_picture('HeightWeight.csv',model_in,np_arr,path)

        return render_template('index.html', href=path)
    

def make_picture(training_data_filename, model, new_inp_np_arr, output_file):
  data= pd.read_csv(training_data_filename)
  heights=data["Height(Inches)"]
  weights=data["Weight(Pounds)"]
  x_new= np.array(list(range(60,76))).reshape(16,1)
  preds= model.predict(x_new)
  fig= px.scatter(x=heights, y=weights, title='Height vs Weight of People',labels={'x': 'Heights (inches)', 'y':'Weight(inches)'})

  fig.add_trace(go.Scatter(x=x_new.reshape(16),y=preds,mode='lines',name='model'))
  new_preds= model.predict(new_inp_np_arr)
  fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)),y=new_preds, name='New Outputs', mode='markers',marker=dict(
            color='purple',
            size=20,
            line=dict(
                color='purple',
                width=2
            ))))
  fig.write_image(output_file, width=800, engine='kaleido')
  fig.show()

def floats_string_to_np_arr(floats_str):
  def is_float(s):
    try:
        float(s)
        return True
    except:
        return False
  floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
  return floats.reshape(len(floats),1)