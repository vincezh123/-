#%%
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, precision_score, \
    recall_score, confusion_matrix, roc_curve, precision_recall_curve
# %%
# import sklearn
# from sklearn.svm import SVC   ##Support Vector Modifier是一种支持向量机的分类器模型。它有许多可调节的参数，
# from sklearn.linear_model import LogisticRegression ##罗辑回归
# from sklearn.ensemble import RandomForestClassifier ##随机森林
# from sklearn.preprocessing import LabelEncoder ##对数据进行编码标签eg.小狗为01小猫02 so on
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay
# from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from statsmodels.sandbox.distributions.gof_new import bootstrap


##remember to run pip freeze > requirements.txt in terminal to generate requirements.txt!!!
#%%
def main():
    st.title('Binary Classification Web App') ##
    st.sidebar.title("Binary Classification Web App")##添加一个侧面的菜单
    st.sidebar.markdown("Are your mushrooms edible or poisonous?🍄")
    st.markdown("Are your mushrooms edible or poisonous?🍄")

    @st.cache_data(persist=True)  ##use cached output unless argument changed
    def load_data():
        data=pd.read_csv("/Users/vince/Desktop/机器学习项目/mushroomClass/mushrooms.csv")
        label=LabelEncoder() ##
        for col in data.columns:
            data[col]=label.fit_transform(data[col])
        return data

    @st.cache_data(persist=True)
    def split(df):
        y=df.type
        x=df.drop(columns=['type'])
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
        return x_train,x_test,y_train,y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
            fig, ax = plt.subplots()  # 创建一个新的 Figure 和 Axes
            disp.plot(ax=ax)  # 将 Axes 传递给 plot 方法
            st.pyplot(fig)
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label="ROC Curve")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            st.pyplot(fig)         ###st.pyplot() deprecated. add fig to avoid raising error

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label="Precision-Recall Curve")
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            st.pyplot(fig)



    df=load_data()
    x_train,x_test,y_train,y_test=split(df)
    class_names=['edible','poisonous']
    # if st.sidebar.checkbox("Show raw data",False):
    #     st.subheader("Mushroom Data Set (Classification）")
    #     st.write(df)  ## 点击sideBar 中的show raw data 展示数据集数据
    st.sidebar.subheader("Choose Classifier")
    classifier=st.sidebar.selectbox("Classifier",("Support Vector Machine","Logistic Regression","Random Forest"))##let user to choose 3 options  from the menu

    if classifier=="Support Vector Machine":
        st.sidebar.subheader("Support Vector Machine Hyperparameters")
        C= st.sidebar.number_input("C (Regularization parameter)",0.00,10.0,step=0.01,key='C ') ## range from 0.01 to 10, step 0.01. key is the widget
        kernel=st.sidebar.radio("Kernel",("linear","rbf"),key='kernel')
        gamma=st.sidebar.radio('Gamma(Kernel Coefficient)',("scale","auto"),key='gamma')
        metrics=st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("Support Vector Machine(SVM) Results")
            model=SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(x_train,y_train)
            accuracy= model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: ",round(accuracy,2))
            st.write("Precision: ",round(precision_score(y_test,y_pred,labels=class_names),2))
            st.write("Recall: ",round(recall_score(y_test,y_pred,labels=class_names),2))
            plot_metrics(metrics)
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameter")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01,
                                    key='C_LR ')  ## range from 0.01 to 10, step 0.01. key is the widget
        max_iter=st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter') ##随机森林的重复次数100-500

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, labels=class_names), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, labels=class_names), 2))
            plot_metrics(metrics)
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameter")
        n_estimators=st.sidebar.number_input("Number of trees in the forest", 100, 5000, step=10, key='n_estimators') ##ask user to add the number of the trees in the forest
        max_depth=st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap_str=st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        bootstrap = True if bootstrap_str == 'True' else False
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Result")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,n_jobs=-1) ##n_job用多少CPU核进行运算-1:100%
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, labels=class_names), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, labels=class_names), 2))
            plot_metrics(metrics)


if __name__ == '__main__':
    main()


###streamlit run .../mushroomClassification.py to start the APP.
