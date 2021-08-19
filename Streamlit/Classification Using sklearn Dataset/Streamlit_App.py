import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from streamlit.config import _gather_usage_stats



def load_data(name):
    data = None
    target_labels = None
    if name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
        target_labels = list(datasets.load_breast_cancer().target_names)
        description = datasets.load_breast_cancer().DESCR
    elif name == 'Iris':
        data = datasets.load_iris()
        target_labels = list(datasets.load_iris().target_names)
        description = datasets.load_iris().DESCR
    elif name == 'Wine':
        data = datasets.load_wine()
        target_labels = list(datasets.load_wine().target_names)
        description = datasets.load_wine().DESCR
    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=['y'])
    df = pd.concat([x, y], axis=1)
    
    return(df, description, target_labels)

def main(df):
    image = Image.open('streamlit.jpg')
    st.image(image, use_column_width=True)

    # Set Title
    st.title('Streamlit Data Science App')

    # Set Sub-title
    st.write('### This is a simple Streamlit app for utilizing different classification algorithms.\
    The data included are classification datasets available in the Scikit-Learn library.\
    This app utilizes a variety of classification algorithms provided by Scikit-Learn,\
    with adjustable hyper-parameters for real-time comparison of accuracy. It also provides feature selection and standardization of the dataset.')

    st.markdown(f'* ## **Current Dataset:** {dataset_name}')
    st.markdown('* ## **Dataset Description**')
    st.write(description)


def eda(df):
    st.title('Exploratory Data Analysis')
    st.write('## Dataframe',df)
    if st.checkbox('Display Shape'):
        st.write('Rows:', df.shape[0],'Columns:', df.shape[1])
    if st.checkbox('Display Columns'):
        st.write(pd.Series(df.columns, name='Column Name'))
    if st.checkbox('Display Summary'):
        st.write(df.describe().T)
    if st.checkbox('Display Null Values'):
        st.write(pd.Series(df.isnull().sum(), name='Total Null'))
    if st.checkbox('Display Data Type'):
        st.dataframe(pd.DataFrame([f"{i}: {df[i].dtype}" for i in df.columns]))
    
        #st.write(df2.dtypes)
    if st.checkbox('Display Correlation Matrix'):
        st.write(df.corr())

def visualization(df):
    viz_type = st.sidebar.radio('Select Visualization Type', ['Scatter Plot', 'Box Plot', 'Histogram', 'Bar Graph', 'Correlation Matrix'])
    st.title('Visualization')
    st.write('## Dataframe', df)
    if viz_type == 'Scatter Plot':
        st.title('Scatter Plot')
        x = st.selectbox('Select x Variable', df.columns, key='s1')
        y = st.selectbox('Select x Variable', df.columns, key='s2')
        if st.checkbox('Add hue'):
            hue = st.selectbox('Select hue Variable', df.columns, key='s3')
        else:
            hue = None
        if st.checkbox('Add line of best fit'):
            reg_line = True
        else:
            reg_line = False
        lmplot = sns.lmplot(x=x, y=y, data=df, hue=hue, fit_reg=reg_line)
        plt.title(f"{x} / {y}")
        st.pyplot(lmplot)
    elif viz_type == 'Box Plot':
        st.title('Box Plot')
        x = st.multiselect('Select variables for visualization', df.columns)
        if st.checkbox('Vertical Orientation'):
            orient = 'v'
        else:
            orient = 'h'
        if st.checkbox('Change Outlier Formula Constant'):
            outliers = st.slider('Outlier Formula Constant (Default=1.5)', min_value=0.0, max_value=2.0, value=1.5, step=0.05)
        else:
            outliers = 1.5
        for var in x:
            fig, ax = plt.subplots()
            sns.boxplot(x=x, data=df, ax=ax, orient=orient, whis=outliers)
            ax.set_title(var)
            st.pyplot(fig)
    elif viz_type == 'Histogram':
        st.title('Histogram')
        x = st.multiselect('Select variables for visualization', df.columns)
        if st.checkbox('Display Histogram', value=True):
            hist = True
        else:
            hist = False
        if st.checkbox('Plot Kernel Density Estimate'):
            kde = True
        else:
            kde = False
        if st.checkbox('Number of Bins'):
            bins = st.slider('Select number of bins', min_value=1, max_value=100, value=50, step=1)
        else:
            bins = None
        if st.checkbox('Add Rug Plot'):
            rug = True
        else:
            rug = False
        for var in x:
            fig, ax = plt.subplots()
            sns.distplot(a=df[x], ax=ax, kde=kde, bins=bins, rug=rug, hist=hist)
            ax.set_title(var)
            st.pyplot(fig)
    elif viz_type == 'Bar Graph':
        st.title('Bar Graph')
        x = st.selectbox('Select column to group by:', df.columns)
        fig, ax = plt.subplots()
        sns.countplot(x=x, data=df, ax=ax)
        ax.set_xticks([np.round(i) for i in np.linspace(0, df[x].nunique(), 11)])
        st.pyplot(fig)
    elif viz_type == 'Correlation Matrix':
        st.title('Correlation Matrix')
        col_option = st.radio('Dataset', ['All Columns', 'Choose Columns'], key='r4')
        annotate = st.radio('Annotate', [True, False])
        if col_option == 'Choose Columns':
            x_col = st.multiselect('Choose x Variable(s)', df.columns, key='m4')
        elif col_option == 'All Columns':
            x_col = df.columns
        if st.checkbox('Change Figure Size'):
            w_1 = st.slider('Width', min_value=1, max_value=20, value=8)
            h_1 = st.slider('Height', min_value=1, max_value=20, value=6)
            if st.button('Apply'):
                w = w_1
                h = h_1
        else:
            w = 8
            h = 6

        try:
            fig, ax = plt.subplots(figsize=(w, h))
            sns.heatmap(df[x_col].corr(), annot=annotate, ax=ax)
            st.pyplot(fig)
        except:
            pass

def model(df):
    st.title('Building a Model')
    st.sidebar.write('='*30)
    classifiers = ['Support Vector Machine', 'K Nearest Neighbors', 'Logistic Regression', 'Random Forest', 'Decision Tree', 'Naive Bayes']
    classifier_name = st.sidebar.selectbox('Select Classifier', classifiers)
    test_size = st.sidebar.slider('Test Size Percentage of Total Dataset', min_value=10, max_value=99, value=20, step=10) / 100
    if st.sidebar.checkbox('Enable Random State'):
        random_state = st.sidebar.slider('Select seed number:', 0, 100)
    else:
        random_state = None
    if st.checkbox('Manual Feature Selection'):
        action = st.radio('Select action:', ['Keep columns', 'Remove columns'])
        x_columns = st.multiselect('Select all columns:', df.drop('y', axis=1).columns)
        if st.button('Apply'):
            if action == 'Keep columns':
                X = df[x_columns]
            elif action == 'Remove columns':
                X = df.drop(x_columns + list('y'), axis=1)
        else:
            X = df.drop('y', axis=1) 
            
    else:
        X = df.drop('y', axis=1)    
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if st.sidebar.checkbox('Standardize Data'):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(pd.DataFrame(X_train, columns=X_train.columns))
        X_test = scaler.transform(pd.DataFrame(X_test, columns=X_test.columns))
        st.success('Data has been standardized using StandardScaler.')
    st.write('## X_train:', X_train)
    st.write('X_train Shape:', X_train.shape)
    st.write('y_train Shape:', y_train.shape)
    st.write('X_test Shape:', X_test.shape)
    st.write('y_test Shape:', y_test.shape)
    
    parameters = dict()
    if classifier_name == 'Support Vector Machine':
        C = st.sidebar.slider('C', 0.1, 15.0)
        kernel = st.sidebar.radio('Kernel', ['linear', 'rbf'])
        gamma = st.sidebar.slider('Gamma', 0.01, 15.0)
        parameters['C'] = C
        parameters['kernel'] = kernel
        parameters['gamma'] = gamma
        parameters['random_state'] = random_state
        clf = SVC(**parameters)
    elif classifier_name == 'K Nearest Neighbors':
        neighbors = st.sidebar.slider('n_neighbors', 1, 15)
        leaf_size = st.sidebar.slider('leaf_size', 1, 50)
        weights = st.sidebar.radio('Weights', ['uniform', 'distance'])
        algorithm = st.sidebar.radio('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        parameters['n_neighbors'] = neighbors
        parameters['leaf_size'] = leaf_size
        parameters['weights'] = weights
        parameters['algorithm'] = algorithm
        clf = KNeighborsClassifier(**parameters)
    elif classifier_name == 'Logistic Regression':
        solver = st.sidebar.radio('Solver', ['liblinear', 'lbfgs', 'newton-cg', 'sag'])
        C = st.sidebar.slider('C', 0.1, 10.0)
        parameters['solver'] = solver
        parameters['C'] = C
        clf = LogisticRegression(**parameters)
    elif classifier_name == 'Random Forest':
        estimators = st.sidebar.slider('n_estimators', 5, 30)
        min_samples_leaf = st.sidebar.slider('min_samples_leaf', 1, 30)
        criterion = st.sidebar.radio('criterion', ['gini', 'entropy'])
        parameters['n_estimators'] = estimators
        parameters['criterion'] = criterion
        parameters['min_samples_leaf'] = min_samples_leaf
        parameters['random_state'] = random_state
        clf = RandomForestClassifier(**parameters)
    elif classifier_name == 'Decision Tree':
        criterion = st.sidebar.radio('Criterion', ['gini', 'entropy'])
        min_samples_split = st.sidebar.slider('Minimum Samples Split', 2, 20)
        min_samples_leaf = st.sidebar.slider('Minimum Samples Leaf', 1, 20)
        parameters['criterion'] = criterion
        parameters['min_samples_split'] = min_samples_split
        parameters['min_samples_leaf'] = min_samples_leaf
        parameters['random_state'] = random_state
        clf = DecisionTreeClassifier(**parameters)
    elif classifier_name == 'Naive Bayes':
        clf = GaussianNB()
    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)
    st.write('Classifier Parameters:', clf.get_params())
    st.write('# Results of', classifier_name)
    st.write('## Model Performance')
    st.write('Accuracy Score(train):', clf.score(X_train, y_train))
    st.write('Accuracy Score(test)', clf.score(X_test, y_test))
    st.write('## Classification Report')
    st.dataframe(classification_report(y_test, clf_pred, output_dict=True))
    st.write('## Confusion Matrix')
    if st.checkbox('Display in heatmap form'):
        fig = plt.figure()
        sns.heatmap(pd.DataFrame(confusion_matrix(y_test, clf_pred),columns=[f'Predicted {target_labels[i]}' for i in range(len(target_labels))],index=[f'Actual {target_labels[i]}' for i in range(len(target_labels))]), annot=True, cmap='Greens', center=0, cbar=False)
        st.pyplot(fig)
    else:
        st.write(pd.DataFrame(confusion_matrix(y_test, clf_pred),columns=[f'Predicted {target_labels[i]}' for i in range(len(target_labels))],index=[f'Actual {target_labels[i]}' for i in range(len(target_labels))]))

    

dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine'))
st.sidebar.write('='*30)

df, description, target_labels = load_data(dataset_name)

activities = ['Main', 'Exploratory Data Analysis', 'Visualization', 'Model']
option = st.sidebar.radio('Select Option', activities)

if option == 'Main':
    main(df)
elif option == 'Exploratory Data Analysis':
    eda(df)
elif option == 'Visualization':
    visualization(df)
elif option == 'Model':
    model(df)







