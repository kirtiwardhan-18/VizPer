
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



#This function manages the file uploads and then saves the files to the disk.
def file_catcher():
    image_file = st.file_uploader("Upload a Dataset",type=['csv'])
    if image_file is not None:
        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        with open("/home/kirti/Desktop/Streamlit/Data-Visualizer/data1.csv","wb") as f: 
            f.write(image_file.getbuffer())         
            st.success("Success")

#The title of our Project
st.title('Data Visualiser')

file_catcher()

#Reading and manipulating data to eliminate errors.
df = pd.read_csv("/home/kirti/Desktop/Streamlit/Data-Visualizer/data1.csv")
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('\n', '_')
df.columns = df.columns.str.replace('\t', '_')

task = st.sidebar.selectbox("What do you want?", ["Intro","Data Plotting","Data Prediction"])
if task == "Intro":
    st.write("This page plots as well as predicts the values using your dataset.")
    st.write("So go on and upload your dataset and then select what you want to do with it using the sidebar on the right.") 
    st.write("**Happy Visualising!** :sunglasses:")
elif task == "Data Plotting":
    #Adding sidebar elements for index column, column1 and column2.
    ind = st.sidebar.selectbox("Select Index Column (For a bit of color):", list(df.columns))
    df.set_index(ind)
    col1 = st.sidebar.selectbox("Column 1:", list(df.columns))
    if not col1:
        st.write("Select a column")
    col2 = st.sidebar.selectbox("Column 2:", list(df.columns))
    if not col2:
        st.write("Select one more column")

    #Generating the plot based on the type of plot selected.
    try:
        plots = ['Area','Bar','Circle','Geoshape','Image','Line','Point','Rectangle','Rule','Square','Text','Tick']
        selplot = st.sidebar.selectbox("Select the type of plot", plots)
        if selplot=='Area':
            chart=alt.Chart(df).mark_area(filled=True).encode(x=(col1), y=(col2), color=ind)
        elif selplot=='Bar':
            chart=alt.Chart(df).mark_bar(filled=True).encode(x=(col1), y=(col2), color=ind)
        elif selplot=='Circle':
            chart=alt.Chart(df).mark_circle(filled=True).encode(x=(col1), y=(col2), color=ind)
        elif selplot=='Geoshape':
            chart=alt.Chart(df).mark_geoshape(filled=True).encode(x=(col1), y=(col2), color=ind)
        elif selplot=='Image':
            chart=alt.Chart(df).mark_image(filled=True).encode(x=(col1), y=(col2), color=ind)
        elif selplot=='Line':
            chart=alt.Chart(df).mark_line(filled=True).encode(x=(col1), y=(col2), color=ind)
        elif selplot=='Point':
            chart=alt.Chart(df).mark_point(filled=True).encode(x=(col1), y=(col2), color=ind)
        elif selplot=='Rectangle':
            chart=alt.Chart(df).mark_rect(filled=True).encode(x=(col1), y=(col2), color=ind)
        elif selplot=='Rule':
            chart=alt.Chart(df).mark_rule(filled=True).encode(x=(col1), y=(col2), color=ind)
        elif selplot=='Square':
            chart=alt.Chart(df).mark_square(filled=True).encode(x=(col1), y=(col2), color=ind)
        elif selplot=='Text':
            chart=alt.Chart(df).mark_text(filled=True).encode(x=(col1), y=(col2), color=ind)
        elif selplot=='Tick':
            chart=alt.Chart(df).mark_tick(filled=True).encode(x=(col1), y=(col2), color=ind)
    
        #Simple button that on clicking displays the plot.
        submit = st.sidebar.button(label='Go')
        if submit:
            st.write("**Plot:**")
            st.altair_chart(chart, use_container_width=True)
    except:
        pass



elif task == "Data Prediction":
    features = st.sidebar.multiselect("Please select all the relevant features:", list(df.columns))
    df.fillna(df.mean(), inplace=True)
    df = df.reset_index(drop=True)
    target = st.sidebar.selectbox("Please select the target:", list(df.columns))
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    for i in features:
        if isinstance(df[i].iloc[0], str):
            df[i] = le.fit_transform(df[i])
        
    #df[target] = le.fit_transform(df[target])
    y = df[target]
    st.write(df[features])
    x=np.empty((len(y),len(features)))
    for p in range(len(features)):
        for q in range(len(y)):
            d=df[features[p]]
            x[q][p]=d[q]    

    np.unique(x,axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 5)
    x_new=[]
    for i in features:
        a=st.sidebar.number_input("Enter "+i)
        x_new.append(a)
    x_new=[x_new]

    try:
        modeltype=st.sidebar.selectbox("Please select a model type:", ["Logistic Regression","Decision Tree","Random Forest","KNeighbors Classifier"])
        if modeltype == "Logistic Regression":
            model = LogisticRegression()

        elif modeltype == "Decision Tree":
            model = DecisionTreeClassifier()

        elif modeltype == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)

        elif modeltype == "KNeighbors Classifier":
            model = KNeighborsClassifier(n_neighbors=1)
    
        submit = st.sidebar.button(label='Go')
        if submit:
            model.fit(x_train,y_train)
            predictions = model.predict(x_test)
            st.write("Accuracy: {}".format(accuracy_score(y_test, predictions)))
            predictions = model.predict(x_new)
            st.write("Prediction:")
            st.write(predictions)
    
    except:
        pass